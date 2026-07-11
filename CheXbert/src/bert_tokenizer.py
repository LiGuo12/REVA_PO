import pandas as pd
from transformers import BertTokenizer, AutoTokenizer
import json
from tqdm import tqdm
import argparse

def get_impressions_from_csv(path):	
        df = pd.read_csv(path)
        imp = df['Report Impression']
        imp = imp.str.strip()
        imp = imp.replace('\n',' ', regex=True)
        imp = imp.replace('\s+', ' ', regex=True)
        imp = imp.str.strip()
        return imp

# def tokenize(impressions, tokenizer):
#         new_impressions = []
#         print("\nTokenizing report impressions. All reports are cut off at 512 tokens.")
#         for i in tqdm(range(impressions.shape[0])):
#                 tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
#                 if tokenized_imp: #not an empty report
#                         res = tokenizer.encode_plus(tokenized_imp)['input_ids']
#                         # res = tokenizer(tokenized_imp, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]

#                         if len(res) > 512: #length exceeds maximum size
#                                 #print("report length bigger than 512")
#                                 res = res[:511] + [tokenizer.sep_token_id]
#                         new_impressions.append(res)
#                 else: #an empty report
#                         new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id]) 
#         return new_impressions

def tokenize(impressions, tokenizer, max_len=512):
    new_impressions = []
    print(f"\nTokenizing report impressions. All reports are cut off at {max_len} tokens.")
    for i in tqdm(range(impressions.shape[0])):
        text = impressions.iloc[i]

        if isinstance(text, str) and text.strip():
            enc = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_len,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            res = enc["input_ids"]
            # Make sure the last one is [SEP]
            if len(res) == max_len and res[-1] != tokenizer.sep_token_id:
                res[-1] = tokenizer.sep_token_id
            new_impressions.append(res)
        else:
            new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id])

    return new_impressions


def load_list(path):
        with open(path, 'r') as filehandle:
                impressions = json.load(filehandle)
                return impressions

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Tokenize radiology report impressions and save as a list.')
        parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                            help='path to csv containing reports. The reports should be \
                            under the \"Report Impression\" column')
        parser.add_argument('-o', '--output_path', type=str, nargs='?', required=True,
                            help='path to intended output file')
        args = parser.parse_args()
        csv_path = args.data
        out_path = args.output_path
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        impressions = get_impressions_from_csv(csv_path)
        new_impressions = tokenize(impressions, tokenizer)
        with open(out_path, 'w') as filehandle:
                json.dump(new_impressions, filehandle)

# conda activate visualchexbert_py38