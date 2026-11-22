"""Turn structured data (countries, periodic table, trivia) into many Q/A phrasings."""
import csv, json, os


def _n(x):
    try:
        return f"{int(float(x)):,}"
    except Exception:
        return None


def countries(path="data/facts/countries.csv"):
    P, E = [], []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        c = r["name"]; cap = r["capital"]; E.append((c, "country"))
        if cap:
            for q in (f"what is the capital of {c}", f"capital of {c}", f"what's the capital of {c}", f"which city is the capital of {c}"):
                P.append((q, f"The capital of {c} is {cap}."))
            P.append((f"{cap} is the capital of which country", f"{cap} is the capital of {c}."))
        if r["currency_name"]:
            for q in (f"what is the currency of {c}", f"what currency does {c} use", f"currency of {c}"):
                P.append((q, f"The currency of {c} is the {r['currency_name']} ({r['currency']})."))
        if r["region"]:
            sub = f", in {r['subregion']}" if r["subregion"] else ""
            for q in (f"which continent is {c} in", f"where is {c}", f"what continent is {c} on", f"where is {c} located"):
                P.append((q, f"{c} is in {r['region']}{sub}."))
        pop = _n(r["population"])
        if pop:
            for q in (f"what is the population of {c}", f"how many people live in {c}", f"population of {c}"):
                P.append((q, f"{c} has a population of about {pop}."))
        if r["phonecode"]:
            P.append((f"what is the phone code of {c}", f"The international dialing code of {c} is +{r['phonecode'].lstrip('+')}."))
        if r["nationality"]:
            P.append((f"what do you call a person from {c}", f"A person from {c} is {r['nationality']}."))
    return P, E


def elements(path="data/facts/elements.csv"):
    P, E = [], []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        el, sy, z = r["Element"], r["Symbol"], r["AtomicNumber"]
        E.append((el, "element"))
        for q in (f"what is the symbol for {el}", f"what is the chemical symbol for {el}", f"chemical symbol of {el}"):
            P.append((q, f"The chemical symbol for {el} is {sy}."))
        P.append((f"which element has the symbol {sy}", f"{sy} is the symbol for {el}."))
        for q in (f"what is the atomic number of {el}", f"atomic number of {el}"):
            P.append((q, f"The atomic number of {el} is {z}."))
        P.append((f"which element has atomic number {z}", f"Element number {z} is {el}."))
        if r["AtomicMass"]:
            P.append((f"what is the atomic mass of {el}", f"The atomic mass of {el} is about {r['AtomicMass']}."))
        if r["Phase"]:
            P.append((f"is {el} a solid liquid or gas", f"{el} is a {r['Phase']} at room temperature."))
        if r["Discoverer"] and r["Year"]:
            P.append((f"who discovered {el}", f"{el} was discovered by {r['Discoverer']} in {r['Year']}."))
    return P, E


def trivia(path="data/facts/trivia.json"):
    return [(d["question"], d["answer"] if d["answer"].endswith(".") else d["answer"] + ".") for d in json.load(open(path))], []


def all_facts():
    P, E = [], []
    for fn in (countries, elements, trivia):
        if os.path.exists(f"data/facts/{fn.__name__ if fn is not elements else 'elements'}.csv") or fn is trivia or True:
            try:
                p, e = fn(); P += p; E += e
            except FileNotFoundError:
                pass
    return P, E
