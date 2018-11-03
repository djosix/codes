defaults = {
    'input_len': 24,
    'input_dim': 4096,
    'embed_dim': 300,
    'embed_train': True,
    'embed_init': None,
    'cnn_filters': [(2, 128), (3, 128), (4, 128)],
    'cnn_merge_dim': 256,
    'cnn_merge_act': 'relu',
    'cnn_dropout': 0.5,
    'rnn_layers': [(128, 0.5)],
    'output_dim': 5,
    'model_options': {
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy'}
}
