class Console:
    def print(self, *args, **kwargs):
        text = " ".join(str(a) for a in args)
        import re
        text = re.sub(r'\[/?[^\]]*\]', '', text)
        print(text)
    def rule(self, text=""):
        import re
        text = re.sub(r'\[/?[^\]]*\]', '', text)
        print(f"\n{'='*60}\n{text}\n{'='*60}")
