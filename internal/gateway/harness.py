# Runs one candidate function on a list of test calls and prints what each call did, as one line
# of JSON. Run by the gateway inside a sandbox: argv = candidate file, inputs file, function name,
# context file (code to run first: imports and helpers the candidate relies on).
import copy, itertools, json, math, os, signal, sys

cand, inputs_path, name, ctx_path = sys.argv[1:5]
out = sys.stdout
sys.stdout = open(os.devnull, "w")  # a candidate that prints must not corrupt the result line


def norm(v):
    """A value as comparable text: floats rounded, sets and dicts in a fixed order."""
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, float):
        return "nan" if math.isnan(v) else repr(round(v, 9))
    if isinstance(v, (set, frozenset)):
        return "{" + ", ".join(sorted(norm(x) for x in v)) + "}"
    if isinstance(v, dict):
        return "{" + ", ".join(sorted(norm(k) + ": " + norm(x) for k, x in v.items())) + "}"
    if isinstance(v, list):
        return "[" + ", ".join(norm(x) for x in v) + "]"
    if isinstance(v, tuple):
        return "(" + ", ".join(norm(x) for x in v) + ")"
    if hasattr(v, "__next__"):
        return norm(list(itertools.islice(v, 1000)))
    r = repr(v)
    return r if len(r) <= 2000 else r[:2000] + "..."


def finish(result):
    out.write(json.dumps(result) + "\n")
    out.flush()
    os._exit(0)


ns = {"__name__": "candidate"}
try:
    exec(compile(open(ctx_path).read(), "context", "exec"), ns)
    exec(compile(open(cand).read(), "candidate", "exec"), ns)
    fn = ns[name]
except KeyError:
    finish({"error": "does not define " + name})
except BaseException as e:
    finish({"error": type(e).__name__ + ": " + str(e)[:200]})


class TimedOut(BaseException):
    pass


def on_alarm(*_):
    raise TimedOut()


signal.signal(signal.SIGALRM, on_alarm)
calls, outcomes = [], []
for args in json.load(open(inputs_path)):
    calls.append(name + "(" + ", ".join(repr(a) for a in args) + ")")
    try:
        signal.setitimer(signal.ITIMER_REAL, 2.0)
        result = fn(*copy.deepcopy(args))
        text = norm(result)
        signal.setitimer(signal.ITIMER_REAL, 0)
        outcomes.append(text)
    except TimedOut:
        outcomes.append("timed out")
    except BaseException as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        outcomes.append("raises " + type(e).__name__)
finish({"calls": calls, "outcomes": outcomes})
