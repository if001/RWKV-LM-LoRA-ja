from argparse import ArgumentParser

def argparse_for_run(def_args):
    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--n_layer", default=32, type=int)
    parser.add_argument("--n_embd", default=4096, type=int)
    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--vocab_size", default="", type=int)

    parser.add_argument("--run_device", default="cuda", type=str)

    parser.add_argument("--model_lora", default="", type=str)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)

    parser.add_argument("--context", default=32, type=str)

    args = parser.parse_args()

    def_args.MODEL_NAME = args.load_model
    def_args.n_layer = args.n_layer
    def_args.n_embd = args.n_embd
    def_args.ctx_len = args.ctx_len
    def_args.vocab_size = args.vocab_size

    def_args.RUN_DEVICE = args.run_device

    def_args.MODEL_LORA = args.model_lora
    def_args.lora_r = args.lora_r
    def_args.lora_alpha = args.lora_alpha

    def_args.context = args.context
    return def_args

    