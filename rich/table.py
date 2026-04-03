class Table:
    def __init__(self, **kwargs):
        self.title = kwargs.get("title", "")
        self.columns = []
        self.rows = []
    def add_column(self, name, **kwargs):
        self.columns.append(name)
    def add_row(self, *args):
        self.rows.append(args)
    def __str__(self):
        import re
        lines = [self.title, " | ".join(self.columns)]
        for r in self.rows:
            lines.append(" | ".join(re.sub(r'\[/?[^\]]*\]', '', str(c)) for c in r))
        return "\n".join(lines)
