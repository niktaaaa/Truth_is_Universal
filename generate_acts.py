import torch as t
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import argparse
import pandas as pd
from tqdm import tqdm
import os
import configparser
import glob
from loguru import logger

config = configparser.ConfigParser()
config.read('config.ini')
logger.info("Configuration file loaded.")

class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        self.out, _ = module_outputs
        logger.debug(f"Hook called on module: {module}, inputs: {module_inputs}, outputs: {module_outputs}")

def load_model(model_family: str, model_size: str, model_type: str, device: str):
    model_path = os.path.join(config[model_family]['weights_directory'], 
                              config[model_family][f'{model_size}_{model_type}_subdir'])
    logger.info(f"Loading model from {model_path}")
    try:
        if model_family == 'Llama2':
            tokenizer = LlamaTokenizer.from_pretrained(str(model_path))
            model = LlamaForCausalLM.from_pretrained(str(model_path))
            tokenizer.bos_token = '<s>'
        else:
            logger.debug(f"Loading Llama3 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            logger.debug(f"Loading Llama3 model")
            model = AutoModelForCausalLM.from_pretrained(str(model_path))
        if model_family == "Gemma2": # Gemma2 requires bfloat16 precision which is only available on new GPUs
            model = model.to(t.bfloat16) # Convert the model to bfloat16 precision
        else:
            model = model.half()  # storing model in float32 precision -> conversion to float16
        logger.info(f"Model {model_family} loaded successfully.")
        logger.debug(f"Model details: {model}")
        return tokenizer, model.to(device)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_statements(dataset_name):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    statements = dataset['statement'].tolist()
    logger.info(f"Loaded {len(statements)} statements from {dataset_name}.csv")
    logger.debug(f"Statements: {statements}")
    return statements

def get_acts(statements, tokenizer, model, layers, device):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.
    """
    # attach hooks
    hooks, handles = [], []
    for layer in layers:
        hook = Hook()
        handle = model.model.layers[layer].register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)
        logger.debug(f"Hook registered for layer {layer}")
    
    # get activations
    acts = {layer : [] for layer in layers}
    logger.info(f"Getting activations for {len(statements)} statements.")
    for statement in tqdm(statements):
        logger.debug(f"Processing statement: {statement}")
        logger.debug(f"acts size: {len(acts)}")
        logger.debug(f"acts memory usage: {sum([act.element_size() for act in acts.values()])}")
        input_ids = tokenizer.encode(statement, return_tensors="pt").to(device)
        logger.debug(f"Input IDs: {input_ids}")
        model(input_ids)
        for layer, hook in zip(layers, hooks):
            acts[layer].append(hook.out[0, -1])
            logger.debug(f"Layer {layer} activation shape: {hook.out[0, -1].shape}")
    
    for layer, act in acts.items():
        acts[layer] = t.stack(act).float()
        logger.info(f"Layer {layer} activations stacked. Shape: {acts[layer].shape}")
        logger.debug(f"Activations: {acts[layer]}")
    
    # remove hooks
    for handle in handles:
        handle.remove()
        logger.debug(f"Hook removed for handle {handle}")
    
    logger.info("Activations collected successfully.")
    return acts

if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    logger.info("Starting activation generation process.")
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--model_family", default="Llama3", help="Model family to use. Options are Llama2, Llama3, Gemma, Gemma2 or Mistral.")
    parser.add_argument("--model_size", default="8B",
                        help="Size of the model to use. Options for Llama3 are 8B or 70B")
    parser.add_argument("--model_type", default="base", help="Whether to choose base or chat model. Options are base or chat.")
    parser.add_argument("--layers", nargs='+', 
                        help="Layers to save embeddings from.")
    parser.add_argument("--datasets", nargs='+',
                        help="Names of datasets, without .csv extension")
    parser.add_argument("--output_dir", default="acts",
                        help="Directory to save activations to.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("-l", "--log-level", default="info", choices=["debug", "info", "warning", "error", "critical"], help="Set the logging level. Options are debug, info, warning, error, critical.")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())
    logger.debug(f"Arguments: {args}")
    
    datasets = args.datasets
    if datasets == ['all_topic_specific']:
        datasets = ['cities', 'sp_en_trans', 'inventors', 'animal_class', 'element_symb', 'facts',
                    'neg_cities', 'neg_sp_en_trans', 'neg_inventors', 'neg_animal_class', 'neg_element_symb', 'neg_facts',
                    'cities_conj', 'sp_en_trans_conj', 'inventors_conj', 'animal_class_conj', 'element_symb_conj', 'facts_conj',
                    'cities_disj', 'sp_en_trans_disj', 'inventors_disj', 'animal_class_disj', 'element_symb_disj', 'facts_disj',
                    'larger_than', 'smaller_than', "cities_de", "neg_cities_de", "sp_en_trans_de", "neg_sp_en_trans_de", "inventors_de", "neg_inventors_de", "animal_class_de",
                  "neg_animal_class_de", "element_symb_de", "neg_element_symb_de", "facts_de", "neg_facts_de"]
    if datasets == ['all']:
        datasets = []
        for file_path in glob.glob('datasets/**/*.csv', recursive=True):
            dataset_name = os.path.relpath(file_path, 'datasets').replace('.csv', '')
            datasets.append(dataset_name)

    t.set_grad_enabled(False)
    tokenizer, model = load_model(args.model_family, args.model_size, args.model_type, args.device)
    
    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")
        statements = load_statements(dataset)
        layers = [int(layer) for layer in args.layers]
        if layers == [-1]:
            layers = list(range(len(model.model.layers)))
        save_dir = f"{args.output_dir}/{args.model_family}/{args.model_size}/{args.model_type}/{dataset}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.info(f"Created directory {save_dir}")

        for idx in range(0, len(statements), 25):
            acts = get_acts(statements[idx:idx + 25], tokenizer, model, layers, args.device)
            for layer, act in acts.items():
                t.save(act, f"{save_dir}/layer_{layer}_{idx}.pt")
                logger.info(f"Saved activations for layer {layer}, batch {idx} to {save_dir}/layer_{layer}_{idx}.pt")
    logger.info("Activation generation process completed.")