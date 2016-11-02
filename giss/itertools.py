def split(iter, key=lambda x : x):
    """Splits an iterator into chunks, according to when the key function changes.
    Yields each chunk as a list."""
    key_fn = key

    records = [next(iter)]
    last_key = key_fn(records[0])
    for item in iter:
        key = key_fn(item)
        if key == last_key:
            records.append(item)
        else:
            yield records
            records = [item]
            last_key = key_fn(item)

    yield records
