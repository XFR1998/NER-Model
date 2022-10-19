from transformers import AdamW, get_linear_schedule_with_warmup

def build_optimizer(args, model, num_total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    print('learning_rate: ', args.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    print('num_training_steps: ', num_total_steps)
    print('warmup_steps: ', num_total_steps*0.06)

    #                                             num_training_steps= args.max_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_total_steps*0.06,
                                                num_training_steps= num_total_steps)
    return optimizer, scheduler


def build_optimizer_diff_lr(args, model, num_total_steps):
    model_lr = {'others': 5e-5, 'bert': 5e-5, 'lstm': 5e-4}
    # model_lr={'others':1e-4, 'roberta':5e-5, 'visual_backbone':8e-6, 'decoder.layers': 9e-5, 'decoder_layer':9e-5, 'convert_swin_base_fc': 11e-5}
    print('----------------采用分层学习率--------------')
    print(model_lr)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    for layer_name in model_lr:
        lr = model_lr[layer_name]
        if layer_name != 'others':  # 设定了特定 lr 的 layer
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
                                                                          and layer_name in n)],
                    "weight_decay": args.weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
                                                                          and layer_name in n)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
        else:  # 其他，默认学习率
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
                                                                          and not any(name in n for name in model_lr))],
                    "weight_decay": args.weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
                                                                          and not any(name in n for name in model_lr))],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
    print('learning_rate: ', args.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    print('num_training_steps: ', num_total_steps)
    print('warmup_steps: ', num_total_steps * 0.06)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_total_steps * 0.06,
                                                num_training_steps=num_total_steps)
    return optimizer, scheduler