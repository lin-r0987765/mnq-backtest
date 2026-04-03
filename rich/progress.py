def track(iterable, description="", **kwargs):
    items = list(iterable)
    total = len(items)
    for i, item in enumerate(items):
        if (i+1) % max(1, total//10) == 0 or i == total-1:
            print(f"  {description} {i+1}/{total}")
        yield item
