#!/usr/bin/env python3

def insert_newlines_every_100_commas(code: str, chunk_size: int = 100) -> str:
    result = []
    stack = []
    comma_count = 0
    inside_brace = False

    for ch in code:
        if ch == '{':
            stack.append('{')
            inside_brace = True
            comma_count = 0
            result.append(ch)
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
            inside_brace = len(stack) > 0
            result.append(ch)
        elif inside_brace and ch == ',':
            comma_count += 1
            result.append(ch)
            if comma_count % chunk_size == 0:
                result.append("\n")
        else:
            result.append(ch)

    return "".join(result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Insert newline every 100 commas inside { } blocks.")
    parser.add_argument("-i", "--input", required=True, help="Input C/C++ file")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    parser.add_argument("--chunk", type=int, default=100, help="Comma chunk size (default=100)")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    new_text = insert_newlines_every_100_commas(text, chunk_size=args.chunk)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(new_text)
