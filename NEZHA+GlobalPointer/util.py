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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_total_steps*0.1,
                                                num_training_steps= num_total_steps)
    return optimizer, scheduler