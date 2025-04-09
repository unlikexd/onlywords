from transformers import AutoModelForCausalLM, AutoTokenizer
from src.onlywords import process
from src.parse_config import parse_config

def main():
    config_path = './config/mistral.yaml'
    config = parse_config(config_path)

    access_token = config['token']

    model_name = config['model']['model_name']

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda",
        token=access_token
    )

    model.to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    res = process(model, tokenizer)
    print(res)

if __name__ == "__main__":
    main()

