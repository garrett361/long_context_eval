import random
import textwrap
import names
import torch
import argparse
import os
import multiprocessing as mp
from tqdm import tqdm

from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random
import datasets

# Modified following: https://arxiv.org/abs/2406.07887 section 3.3.3
class PhoneBookDataset(Dataset):
    def __init__(self,
                 length: int = 1000,
                 tokenizer: AutoTokenizer = None,
                 size: int = 1000,
                 few_shot: int = 2,
                 reversed: bool = False,
                 random_depth: bool = False):

        assert length > 500, "The length must be greater than 500 to ensure proper depth control."

        self.length = length
        self.tokenizer = tokenizer
        self.few_shot = few_shot
        self.size = size
        self.reversed = reversed
        self.random_depth = random_depth

        if not self.reversed:
            self.few_shot_template = textwrap.dedent('''\
                What is the phone number for {name}?
                Answer: {phone}\
                ''')
        else:
            self.prefix_template = textwrap.dedent('''\
                Given a phone book with entries of the form:
                {few_shot_examples}
                ... and so on.
                Please find the phone number for {name}.
                I will ask you for this phone number later.
                Memorize it so you can respond correctly.
                Remember {name}.
                Here is the phonebook:\n
                ''')

            self.suffix_template = textwrap.dedent('''\
                Okay, that is the end of the phonebook.
                Remember when I asked you to memorize the phone number for {name}?
                What is the phone number for {name}?\
                ''')

    def _gen_phone(self):
        # Ensure the area code and exchange don't start with 0 or 1.
        area_code = random.randint(200, 999)
        exchange = random.randint(200, 999)
        subscriber = random.randint(1000, 9999)
        return f"{area_code}-{exchange}-{subscriber}"

    # Generate phone book with controlled insertion of few-shot examples
    def _gen_phone_book(self, depth: int = 50):
        # Generate names and phone numbers for few-shot samples and target.
        name_list = []
        phone_list = []
        for _ in range(self.few_shot + 1):
            name_list.append(names.get_full_name())
            phone_list.append(self._gen_phone())

        if not self.reversed:
            suffix = []
            for i in range(self.few_shot):
                suffix.append(self.few_shot_template.format(name=name_list[i], phone=phone_list[i]))
            suffix.append(self.few_shot_template.format(name=name_list[self.few_shot], phone=''))
            prefix = ""
            suffix = "\n".join(suffix).strip()
        else:
            few_shot_examples = []
            for i in range(self.few_shot):
                few_shot_examples.append(f"{name_list[i]}: {phone_list[i]}")
            few_shot_examples = "\n".join(few_shot_examples)
            prefix = self.prefix_template.format(few_shot_examples=few_shot_examples, name=name_list[self.few_shot])
            suffix = self.suffix_template.format(name=name_list[self.few_shot]).strip()

        # The label is the target phone number (last element).
        label = phone_list[self.few_shot]

        if self.tokenizer:
            prefix_ids = self.tokenizer(prefix, add_special_tokens=(self.tokenizer.bos_token is not None))["input_ids"]
            suffix_ids = self.tokenizer(suffix, add_special_tokens=False)["input_ids"]

            # Reserve space for prefix and suffix.
            phone_book_count = self.length - len(prefix_ids) - len(suffix_ids)
            input_ids = prefix_ids.copy()
        else:
            phone_book_count = self.length

        phone_book_raw = []

        # Pre-calculate positions for insertions.
        few_shot_positions = [((j + 1) * phone_book_count) // (self.few_shot + 1) for j in range(self.few_shot)]
        few_shot_idx = 0
        target_position = int((depth / 100) * phone_book_count * 0.97) #
        target_inserted = False

        i = 0
        while i < phone_book_count:
            # Check if we need to insert the target entry.
            if not target_inserted and i >= target_position:
                phone_book_entry = f"{name_list[self.few_shot]}: {phone_list[self.few_shot]}\n"
                target_inserted = True
            # Otherwise, check if it's time for a few-shot example.
            elif few_shot_idx < self.few_shot and i >= few_shot_positions[few_shot_idx]:
                phone_book_entry = f"{name_list[few_shot_idx]}: {phone_list[few_shot_idx]}\n"
                few_shot_idx += 1
            else:
                # Otherwise, add a random entry.
                phone_book_entry = f"{names.get_full_name()}: {self._gen_phone()}\n"

            if self.tokenizer:
                phone_book_ids = self.tokenizer(phone_book_entry, add_special_tokens=False)["input_ids"]
                if i + len(phone_book_ids) > phone_book_count:
                    break
                i += len(phone_book_ids)
                input_ids.extend(phone_book_ids)
            else:
                i += 1
            phone_book_raw.append(phone_book_entry)


        phone_book_raw = "".join(phone_book_raw)
        input_raw = prefix + phone_book_raw + suffix
        if self.tokenizer:
            input_ids.extend(suffix_ids)
            pad_length = phone_book_count - i
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            if self.tokenizer.padding_side == "right":
                input_ids[len(input_ids):] = [pad_id] * pad_length
                input_ids = torch.tensor(input_ids)
                attention_mask = torch.cat((torch.ones(self.length - pad_length), torch.zeros(pad_length)), dim=0)
            else:
                input_ids[0:0] = [pad_id] * pad_length
                input_ids = torch.tensor(input_ids)
                attention_mask = torch.cat((torch.zeros(pad_length), torch.ones(self.length - pad_length)), dim=0)
        else:
            input_ids, attention_mask = None, None

        return input_raw, input_ids, attention_mask, label

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.random_depth:
            depth = random.random() * 100
        else:
            depth = idx % 100  # convert idx into a percentage evenly
        input_raw, input_ids, attention_mask, label = self._gen_phone_book(depth)
        if self.tokenizer:
            return {
                'input_raw': input_raw,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label,
                'depth': depth
            }
        else:
            return {
                'input_raw': input_raw,
                'label': label,
                'depth': depth
            }

global_dataset = None

def init_worker(model_name, dataset_params):
    global global_dataset

    tokenizer = None
    if model_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    global_dataset = PhoneBookDataset(tokenizer=tokenizer, **dataset_params)

def generate_sample(idx):
    return global_dataset[idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help="Tokenizer for length control.")
    parser.add_argument('--length', default=4096, type=int, help="Tokenized input sequence length.")
    parser.add_argument('--size', default=100, type=int, help="Number of samples in dataset.")
    parser.add_argument('--few-shot', default=2, type=int, help="Few-shot examples in prompt.")
    parser.add_argument('--reversed', action='store_true', help="Use reversed prompt template.")
    parser.add_argument('--random-depth', action='store_true', help="Use random depth for each sample.")
    parser.add_argument('--save-path', required=True, type=str, help="Path to save dataset.")
    args = parser.parse_args()

    torch.cuda.manual_seed(42)
    torch.manual_seed(42)

    dataset_params = {
        'length': args.length,
        'size': args.size,
        'few_shot': args.few_shot,
        'reversed': args.reversed,
        'random_depth': args.random_depth
    }

    # Set up multiprocessing pool and reinitialize tokenizer in each worker.
    with mp.Pool(processes=mp.cpu_count(),
                 initializer=init_worker,
                 initargs=(args.model, dataset_params)) as pool:
        dataset_list = list(tqdm(pool.imap(generate_sample, range(args.size)),
                                 total=args.size, desc="Generating dataset"))

    avg_len = 0.0
    for i in dataset_list:
        avg_len += torch.sum(i['attention_mask'])
    print(f"---Sanity Check: average length = {avg_len/len(dataset_list)}, dataset size = {len(dataset_list)}---")

    dataset = datasets.Dataset.from_list(dataset_list)

    tokenized = (args.model is not None)
    dataset_name = f"length={args.length}_tokenized={tokenized}_size={args.size}_few-shot={args.few_shot}_reversed={args.reversed}"
    if "llama" in args.model.lower():
        model_name = "llama"
    elif "bamba" in args.model.lower():
        model_name = "bamba"
    else:
        model_name = "other"
    dataset_path = os.path.join(args.save_path, model_name)
    os.makedirs(dataset_path, exist_ok=True)
    dataset_path = os.path.join(dataset_path, dataset_name)
    dataset.save_to_disk(dataset_path)
    print(f"Dataset saved to {dataset_path}")
