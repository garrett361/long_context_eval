import os
import gc
import argparse
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import datasets
# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from phone_book_dataset import PhoneBookDataset

import torch
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# FSDP2
from torch.distributed._composable.fsdp import fully_shard, register_fsdp_forward_method
from torch.distributed.device_mesh import init_device_mesh


# See FSDP's problem with model.generate:
# https://github.com/huggingface/transformers/issues/30228#issuecomment-2350022762
# FSDP2 turns out to be a better option

def phone_book_evaluation(model, tokenizer, dataloader, rank, world_size):
    device = torch.device(torch.distributed.get_rank())
    batch_registers = torch.zeros(3).to(device) # [correct, total, length]
    depth_registers = torch.zeros(100).to(device) # Registers success cases for each percentile

    dataloader_pb = tqdm(dataloader, total=len(dataloader)) if rank == 0 else dataloader

    for batch in dataloader_pb:
        inputs = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        labels = batch['label']
        depths = batch['depth']
        max_length = inputs.shape[1] + 50

        out = model.generate(
            input_ids=inputs,
            attention_mask=attention_masks,
            max_length=max_length,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )

        input_length = inputs.shape[1]
        generated_tokens = out.sequences[:, input_length:]
        decoded_preds = tokenizer.batch_decode(generated_tokens.tolist())

        batch_correct = 0
        for pred, label, depth in zip(decoded_preds, labels, depths):
            pred_model = pred.split("\n")[0].strip()
            if pred_model == label:
                batch_correct += 1
                depth_registers[depth] += 1

        batch_registers[0] += batch_correct
        batch_registers[1] += inputs.shape[0]
        # register average actual sequence length
        batch_registers[2] += torch.sum(attention_masks).item()

    dist.all_reduce(batch_registers, op=dist.ReduceOp.SUM)
    dist.all_reduce(depth_registers, op=dist.ReduceOp.SUM)

    if rank == 0:
        total_accuracy = batch_registers[0] / batch_registers[1]
        avg_length = batch_registers[2] / batch_registers[1]
        print('Average length: {:.2f} tokens, Accuracy: {}/{} ({:.2f}%)\n'.format(
            avg_length, batch_registers[0], batch_registers[1],
            100. * total_accuracy))
    return batch_registers, depth_registers


def main(rank, world_size, args):
    if args.wandb:
        import wandb  # type: ignore

        if rank == 0:
            print("--> wandb is enabled!")
            wandb.init(project=args.wandb_project, id=args.wandb_id)
            wandb.config = vars(args)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method="env://")

    device = torch.device(torch.distributed.get_rank())
    torch.cuda.set_device(device)

    config_kwargs = {
        # "max_position_embeddings": 16384,
        # "rope_theta": 40000.0,
        "attn_implementation": "flash_attention_2"
    }
    config = AutoConfig.from_pretrained(args.model, **config_kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.bfloat16,
        config=config).to(device)

    mesh = init_device_mesh("cuda", (torch.distributed.get_world_size(),))

    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    register_fsdp_forward_method(model, "generate")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'




    for length in map(int, args.length_list.split(',')):
        print(f"Begin testing on phone books of length {length}...")
        phonebook = None

        if args.save_path is not None and os.path.exists(args.save_path):
            dataset_name = f"length={length}_tokenized=True_size={args.size}_few-shot={args.few_shot}_reversed={args.reversed}"
            if "llama" in (args.model.lower(),  args.model_type):
                model_name = "llama"
            elif "bamba" in (args.model.lower(),  args.model_type):
                model_name = "bamba"
            else:
                model_name = "other"
            dataset_path = os.path.join(args.save_path, model_name, dataset_name)

            print(f"Looking for dataset in:{dataset_path}")
            if os.path.exists(dataset_path):
                print(f"Load dataset from {dataset_path}")
                phonebook = datasets.load_from_disk(dataset_path).with_format("torch")

        if phonebook is None:
            print(f"Generating a new dataset.")
            phonebook = PhoneBookDataset(
                length=length,
                tokenizer=tokenizer,
                size=args.size,
                few_shot=args.few_shot,
                reversed=args.reversed,
                random_depth=args.random_depth
            )

        sampler = DistributedSampler(phonebook, rank=rank, num_replicas=world_size, shuffle=True)
        dataloader = DataLoader(phonebook, batch_size=args.batch_size, sampler=sampler, num_workers=1)

        batch_registers, depth_registers = phone_book_evaluation(model,tokenizer,dataloader,rank,world_size)
        if rank == 0:
            print(batch_registers, depth_registers)
            if args.wandb:
                total_accuracy = batch_registers[0] / batch_registers[1]
                wandb.log({"acc": total_accuracy.item()}, step=length)
        torch.distributed.barrier()
        gc.collect()
        torch.cuda.empty_cache()

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model-type', type=str)
    parser.add_argument('--length-list', default='4096,8192,16384', type=str, help="List of tokenized input sequence length.")
    parser.add_argument('--batch-size', default=32, type=int, help="Batch size of the input.")
    parser.add_argument('--size', default=100, type=int, help="Number of samples in dataset.")
    parser.add_argument('--few-shot', default=2, type=int, help="Few-shot examples in prompt.")
    parser.add_argument('--reversed', action='store_true', help="Use reversed prompt template.")
    parser.add_argument('--random-depth', action='store_true', help="Use random depth for each sample.")
    parser.add_argument('--save-path', type=str, help="Path to save dataset.")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project',type=str, default=None)
    parser.add_argument('--wandb_id',type=str, default=None)
    parser.add_argument('--tokenizer_path',type=str, default=None)
    args = parser.parse_args()

    torch.cuda.manual_seed(42)
    torch.manual_seed(42)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    main(rank, world_size, args)
