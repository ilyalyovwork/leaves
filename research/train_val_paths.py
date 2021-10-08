

def get_paths(root, format='jpg', paths_count=None):

    path_i = 0
    paths = []
    for path in root.glob(f'**/*.{format}'):
        if path_i == paths_count:
            break

        paths.append(path)
        path_i += 1
    return paths


def get_train_val_paths(root, format='jpg', val_size=0.2, paths_count=None):
    paths = get_paths(root, format=format, paths_count=paths_count)
    if paths_count is None:
        paths_count = len(paths)
    val_count = int(val_size * paths_count)
    train_count = paths_count - val_count
    return paths[:train_count], paths[train_count:]

