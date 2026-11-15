#!/usr/bin/env python3
"""End-to-end test of the gateway's build queue: a 36-function Python library.

It does what Bill the Cat's delegate_build will do, from a terminal:

  1. POST /v1/builds with one task per function.
  2. Poll until every agent has finished and the program is assembled.
  3. Check the program: run each function's test calls and compare with the expected values.
  4. Send every function that failed back through POST /v1/builds/{id}/rework (a five-agent
     vote), and check again. At most two rounds.

Run it with the gateway service up:

    python3 examples/textkit_test.py

Options: --gateway URL (default http://127.0.0.1:8765), --out FILE (default ./textkit_build.py),
--local (send the scope header, so only local models are used), --no-rework.

The assembled program is code written by other models. It is checked in a child process, on
macOS inside sandbox-exec with no network and no file writes.

Cost: one call per function spread over the free agents, plus five calls for each function that
has to be reworked.
"""
import argparse, json, os, subprocess, sys, tempfile, time, urllib.error, urllib.request

# (name, what to write, [(test expression, expected value as a Python literal), ...])
FUNCTIONS = [
    ("slugify", "slugify(s): lowercase s, replace every run of characters other than a-z and 0-9 with one hyphen, and strip hyphens from both ends.",
     [("slugify('Hello, World!')", "'hello-world'"), ("slugify('  A  b ')", "'a-b'"), ("slugify('x')", "'x'")]),
    ("word_count", "word_count(s): return a dict mapping each word to the number of times it occurs. A word is a run of letters, digits or apostrophes; compare case-insensitively and use lowercase keys.",
     [("word_count('The cat the hat')", "{'the': 2, 'cat': 1, 'hat': 1}"), ("word_count('')", "{}")]),
    ("is_palindrome", "is_palindrome(s): True if s reads the same both ways, ignoring case and every character that is not a letter or digit. The empty string is a palindrome.",
     [("is_palindrome('A man, a plan, a canal: Panama')", "True"), ("is_palindrome('hello')", "False"), ("is_palindrome('')", "True")]),
    ("reverse_words", "reverse_words(s): return the words of s in reverse order joined by single spaces. Words are separated by any amount of whitespace.",
     [("reverse_words('the quick  brown')", "'brown quick the'"), ("reverse_words('one')", "'one'")]),
    ("title_case", "title_case(s): for each space-separated word, uppercase its first character and lowercase the rest; keep single spaces between words.",
     [("title_case('hELLO wORLD')", "'Hello World'"), ("title_case('a')", "'A'")]),
    ("camel_to_snake", "camel_to_snake(s): insert an underscore before every uppercase letter that is not the first character, then lowercase the whole string.",
     [("camel_to_snake('camelCaseString')", "'camel_case_string'"), ("camel_to_snake('simple')", "'simple'"), ("camel_to_snake('aB')", "'a_b'")]),
    ("snake_to_camel", "snake_to_camel(s): turn snake_case into camelCase: the first word stays as it is, each later word gets its first letter uppercased, and the underscores are removed.",
     [("snake_to_camel('snake_case_string')", "'snakeCaseString'"), ("snake_to_camel('x')", "'x'")]),
    ("truncate", "truncate(s, n): if len(s) <= n return s unchanged; otherwise return the first n - 3 characters of s followed by '...'. Assume n >= 3.",
     [("truncate('hello world', 8)", "'hello...'"), ("truncate('hi', 8)", "'hi'")]),
    ("count_vowels", "count_vowels(s): the number of characters in s that are one of a, e, i, o, u, in either case.",
     [("count_vowels('Hello World')", "3"), ("count_vowels('xyz')", "0")]),
    ("is_anagram", "is_anagram(a, b): True if a and b use exactly the same letters the same number of times, ignoring case and spaces.",
     [("is_anagram('Listen', 'Silent')", "True"), ("is_anagram('abc', 'abd')", "False"), ("is_anagram('a b', 'ba')", "True")]),
    ("remove_duplicates", "remove_duplicates(lst): return a new list with duplicates removed, keeping the first occurrence of each value in its original order.",
     [("remove_duplicates([3, 1, 3, 2, 1])", "[3, 1, 2]"), ("remove_duplicates([])", "[]")]),
    ("flatten", "flatten(lst): return a flat list of the values in lst, where lst may contain lists nested to any depth.",
     [("flatten([1, [2, [3, [4]]], 5])", "[1, 2, 3, 4, 5]"), ("flatten([])", "[]")]),
    ("chunk", "chunk(lst, n): split lst into consecutive lists of length n; the last one may be shorter. Assume n >= 1.",
     [("chunk([1, 2, 3, 4, 5], 2)", "[[1, 2], [3, 4], [5]]"), ("chunk([], 3)", "[]")]),
    ("fib", "fib(n): the n-th Fibonacci number, with fib(0) == 0 and fib(1) == 1. Assume n >= 0.",
     [("fib(10)", "55"), ("fib(0)", "0"), ("fib(1)", "1")]),
    ("is_prime", "is_prime(n): True if the integer n is a prime number. Numbers below 2 are not prime.",
     [("is_prime(97)", "True"), ("is_prime(1)", "False"), ("is_prime(91)", "False"), ("is_prime(2)", "True")]),
    ("gcd", "gcd(a, b): the greatest common divisor of two non-negative integers, by Euclid's algorithm. gcd(0, x) is x.",
     [("gcd(48, 18)", "6"), ("gcd(0, 5)", "5")]),
    ("lcm", "lcm(a, b): the least common multiple of two non-negative integers. It must call gcd(a, b), a function defined elsewhere in the same file with that exact signature; do not define gcd yourself. lcm(0, x) is 0.",
     [("lcm(4, 6)", "12"), ("lcm(0, 5)", "0")]),
    ("factorial", "factorial(n): n! for an integer n >= 0, with factorial(0) == 1.",
     [("factorial(5)", "120"), ("factorial(0)", "1")]),
    ("prime_factors", "prime_factors(n): the prime factors of the integer n >= 1 as a list in ascending order, each repeated as often as it divides n. prime_factors(1) is [].",
     [("prime_factors(360)", "[2, 2, 2, 3, 3, 5]"), ("prime_factors(13)", "[13]"), ("prime_factors(1)", "[]")]),
    ("roman_to_int", "roman_to_int(s): the integer value of the Roman numeral s (uppercase, standard subtractive notation).",
     [("roman_to_int('MCMXCIV')", "1994"), ("roman_to_int('IX')", "9")]),
    ("int_to_roman", "int_to_roman(n): the Roman numeral for the integer n, 1 <= n <= 3999, in standard subtractive notation.",
     [("int_to_roman(1994)", "'MCMXCIV'"), ("int_to_roman(4)", "'IV'")]),
    ("binary_search", "binary_search(lst, x): lst is sorted ascending; return the index of x in lst, or -1 if it is not there.",
     [("binary_search([1, 3, 5, 7, 9], 7)", "3"), ("binary_search([1, 3, 5], 4)", "-1"), ("binary_search([], 1)", "-1")]),
    ("merge_sorted", "merge_sorted(a, b): a and b are lists sorted ascending; return one sorted list of all their values.",
     [("merge_sorted([1, 4, 9], [2, 3, 10])", "[1, 2, 3, 4, 9, 10]"), ("merge_sorted([], [1])", "[1]")]),
    ("rotate", "rotate(lst, k): return a new list with the values of lst rotated k positions to the right; k may be larger than len(lst); an empty list gives [].",
     [("rotate([1, 2, 3, 4, 5], 2)", "[4, 5, 1, 2, 3]"), ("rotate([1, 2, 3], 4)", "[3, 1, 2]"), ("rotate([], 3)", "[]")]),
    ("caesar_cipher", "caesar_cipher(s, k): shift each ASCII letter k places along the alphabet, wrapping around and keeping its case; k may be negative; other characters are unchanged.",
     [("caesar_cipher('Hello, World!', 3)", "'Khoor, Zruog!'"), ("caesar_cipher('abc', -1)", "'zab'")]),
    ("rle_encode", "rle_encode(s): run-length encode s as each character followed by the decimal length of its run. rle_encode('') is ''.",
     [("rle_encode('aaabbc')", "'a3b2c1'"), ("rle_encode('')", "''")]),
    ("rle_decode", "rle_decode(s): the inverse of run-length encoding: s is a sequence of one non-digit character followed by a decimal count, which may have several digits.",
     [("rle_decode('a3b2c1')", "'aaabbc'"), ("rle_decode('x12')", "'xxxxxxxxxxxx'"), ("rle_decode('')", "''")]),
    ("is_balanced", "is_balanced(s): True if every bracket among ()[]{} in s is closed by the matching bracket in the right order; all other characters are ignored.",
     [("is_balanced('{[()]}')", "True"), ("is_balanced('([)]')", "False"), ("is_balanced('')", "True"), ("is_balanced('(')", "False")]),
    ("to_base", "to_base(n, b): the non-negative integer n written in base b, 2 <= b <= 36, using digits 0-9 then lowercase a-z. to_base(0, b) is '0'.",
     [("to_base(255, 16)", "'ff'"), ("to_base(5, 2)", "'101'"), ("to_base(0, 7)", "'0'")]),
    ("from_base", "from_base(s, b): the integer that the string s denotes in base b, 2 <= b <= 36; digits are 0-9 then a-z in either case.",
     [("from_base('ff', 16)", "255"), ("from_base('101', 2)", "5"), ("from_base('FF', 16)", "255")]),
    ("median", "median(lst): the median of a non-empty list of numbers: the middle value after sorting, or the average of the two middle values when the length is even.",
     [("median([3, 1, 2])", "2"), ("median([4, 1, 3, 2])", "2.5")]),
    ("mode", "mode(lst): the value that occurs most often in a non-empty list of numbers; if several tie, the smallest of them.",
     [("mode([1, 2, 2, 3, 3])", "2"), ("mode([5])", "5")]),
    ("levenshtein", "levenshtein(a, b): the edit distance between strings a and b: the fewest single-character insertions, deletions and substitutions that turn a into b.",
     [("levenshtein('kitten', 'sitting')", "3"), ("levenshtein('', 'abc')", "3"), ("levenshtein('same', 'same')", "0")]),
    ("hamming", "hamming(a, b): the number of positions at which two strings of equal length differ.",
     [("hamming('karolin', 'kathrin')", "3"), ("hamming('', '')", "0")]),
    ("digit_sum", "digit_sum(n): the sum of the decimal digits of the integer n, ignoring its sign.",
     [("digit_sum(12345)", "15"), ("digit_sum(-91)", "10"), ("digit_sum(0)", "0")]),
    ("is_leap_year", "is_leap_year(y): True if year y is a leap year in the Gregorian calendar.",
     [("is_leap_year(2000)", "True"), ("is_leap_year(1900)", "False"), ("is_leap_year(2024)", "True"), ("is_leap_year(2023)", "False")]),
]

# Runs in a child process: load the program, run every test, print the failures as JSON.
CHECKER = r'''
import json, signal, sys
program, spec = sys.argv[1], json.load(open(sys.argv[2]))
out, sys.stdout = sys.stdout, open("/dev/null", "w")
failures, ns = {}, {"__name__": "textkit"}
try:
    exec(compile(open(program).read(), "textkit_build.py", "exec"), ns)
except BaseException as e:
    line = getattr(e, "lineno", None)
    out.write(json.dumps({"__program__": "the program does not load: %s: %s%s" % (type(e).__name__, e, " (line %s)" % line if line else "")}))
    sys.exit(0)
class TimedOut(BaseException): pass
def alarm(*_): raise TimedOut()
signal.signal(signal.SIGALRM, alarm)
for name, tests in spec:
    for expr, expected in tests:
        try:
            signal.setitimer(signal.ITIMER_REAL, 2.0)
            got = eval(expr, ns)
            signal.setitimer(signal.ITIMER_REAL, 0)
            want = eval(expected)
            if got != want or type(got) is not type(want) and not (isinstance(got, (int, float)) and isinstance(want, (int, float))):
                failures.setdefault(name, []).append("%s returned %r, expected %r" % (expr, got, want))
        except TimedOut:
            failures.setdefault(name, []).append("%s did not return within 2 seconds" % expr)
        except BaseException as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            failures.setdefault(name, []).append("%s raised %s: %s" % (expr, type(e).__name__, e))
out.write(json.dumps(failures))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gateway", default=os.environ.get("AI_GATEWAY_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--out", default="textkit_build.py")
    ap.add_argument("--local", action="store_true", help="local models only (sends X-AI-Gateway-Scope: local)")
    ap.add_argument("--no-rework", action="store_true")
    args = ap.parse_args()
    headers = {"Content-Type": "application/json"}
    if args.local:
        headers["X-AI-Gateway-Scope"] = "local"

    def call(method, path, body=None):
        req = urllib.request.Request(args.gateway + path, method=method, headers=headers,
                                     data=None if body is None else json.dumps(body).encode())
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            sys.exit("gateway said %d: %s" % (e.code, e.read().decode(errors="replace")))
        except urllib.error.URLError as e:
            sys.exit("cannot reach the gateway at %s (%s). Is the service running? Try: ai-gateway service status" % (args.gateway, e.reason))

    start = time.time()
    stamp = lambda: "[%4.0fs]" % (time.time() - start)

    def wait(build_id):
        seen = -1
        while True:
            v = call("GET", "/v1/builds/%s?wait=15" % build_id)
            done = sum(p["status"] == "done" for p in v["parts"])
            if v["status"] != "running":
                return v
            if done != seen:
                running = [p["id"] for p in v["parts"] if p["status"] == "running"]
                print("%s %d of %d written; in progress: %s" % (stamp(), done, len(v["parts"]), ", ".join(running) or "-"))
                seen = done

    def check(program):
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in (("program.py", program), ("checker.py", CHECKER), ("spec.json", json.dumps([[n, t] for n, _, t in FUNCTIONS]))):
                with open(os.path.join(tmp, name), "w") as f:
                    f.write(text)
            cmd = [sys.executable, "-I", "checker.py", "program.py", "spec.json"]
            if sys.platform == "darwin" and os.path.exists("/usr/bin/sandbox-exec"):  # model-written code: no network, no writes
                cmd = ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)(deny file-write*)"] + cmd
            try:
                r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=120)
                return json.loads(r.stdout)
            except (subprocess.TimeoutExpired, ValueError) as e:
                return {"__program__": "the checker did not finish: %s" % e}

    def report(v):
        tally = {}
        for p in v["parts"]:
            if p["status"] == "done":
                tally[p["author"]] = tally.get(p["author"], 0) + 1
        print("\nWho wrote how many (fast agents take more from the queue):")
        for author, n in sorted(tally.items(), key=lambda kv: -kv[1]):
            print("  %-22s %2d  %s" % (author, n, "#" * n))
        for note in v["notes"]:
            print("  note: " + note)

    tasks = [{"id": name, "prompt": "Write a Python function " + what, "tests": [expr for expr, _ in tests]} for name, what, tests in FUNCTIONS]
    made = call("POST", "/v1/builds", {"language": "python", "tasks": tasks,
                                       "context": "These functions are parts of one small utility module, textkit. Use only the Python standard library."})
    build_id = made["build_id"]
    print("%s build %s started: %d functions in the queue" % (stamp(), build_id, len(tasks)))
    v = wait(build_id)
    print("%s the queue is empty: status %s" % (stamp(), v["status"]))
    report(v)

    for round_no in range(3):
        with open(args.out, "w") as f:
            f.write(v["program"] + "\n")
        failures = check(v["program"])
        unwritten = {p["id"]: "it was not written: " + (p["error"] or p["status"]) for p in v["parts"] if p["status"] != "done"}
        for name, why in unwritten.items():
            failures.setdefault(name, []).insert(0, why)
        if "__program__" in failures:
            print("\n%s %s" % (stamp(), failures["__program__"]))
            print("The program is in %s. A syntax error in one part stops them all; rework the part on that line." % args.out)
            return 1
        tested = sum(len(t) for _, _, t in FUNCTIONS)
        failed_tests = sum(len(f) for f in failures.values())
        print("\n%s check %d: %d of %d functions pass (%d of %d test calls)" % (stamp(), round_no + 1, len(FUNCTIONS) - len(failures), len(FUNCTIONS), tested - failed_tests, tested))
        for name, why in failures.items():
            print("  FAIL %-18s %s" % (name, why[0]))
        if not failures:
            print("\nAll %d functions pass. The program is in %s (%d lines). Total %.0f s." % (len(FUNCTIONS), args.out, v["program"].count("\n") + 1, time.time() - start))
            return 0
        if args.no_rework or round_no == 2:
            break
        print("\n%s sending %d functions back through a five-agent vote" % (stamp(), len(failures)))
        call("POST", "/v1/builds/%s/rework" % build_id, {"failures": [{"id": n, "error": "\n".join(w)} for n, w in failures.items()]})
        v = wait(build_id)
        report(v)
    print("\n%d functions still fail. The program is in %s. Total %.0f s." % (len(failures), args.out, time.time() - start))
    return 1


if __name__ == "__main__":
    sys.exit(main())
