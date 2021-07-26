import logging
import time

import torch
import transformers
import yaml

import numpy as np

# TODO: Add pretrained weights initialization for all architectures
NAME2MODEL = {
    "bert": transformers.BertModel,
    "roberta": transformers.RobertaModel,
    "deberta": transformers.DebertaModel,
    "distilbert": transformers.DistilBertModel,
    "funnel_transformer": transformers.FunnelModel,
    "ibert": transformers.IBertModel,
    "longformer": transformers.LongformerModel,
    "mobile_bert": transformers.MobileBertModel,
    "reformer": transformers.ReformerModel,
    "squeeze_bert": transformers.SqueezeBertModel,
}


def main(model_name="bert", random_text=False, requires_grad=True, batch_size=1, iters=20,
         use_cuda=False):
    init_time = time.time()
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")
    model = NAME2MODEL[model_name].from_pretrained("bert-base-cased")
    model.requires_grad_(requires_grad)
    model.eval()

    seq_lengths = [2 ** seqlen for seqlen in range(10)]
    for seq_len in seq_lengths:
        time_per_example = []
        for _ in range(iters):
            if random_text:
                input_ids = torch.randint(tokenizer.vocab_size, (seq_len))
            else:
                tokens = seq_len * ["a"]
                input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
            input_ids = input_ids.unsqueeze(0)
            if use_cuda:
                input_ids = input_ids.to("cuda")
                model.to("cuda")

            init_example_time = time.time()
            outputs = model(input_ids)
            time_per_example.append(time.time() - init_example_time)
        print(f"Model: {model_name}, Seqlen: {seq_len} ---- Time per example:"
                    f" {np.mean(time_per_example)}")


if __name__ == "__main__":
    main()
