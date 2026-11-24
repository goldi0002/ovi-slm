"""Training data for ovi.gen: real dialogue (DailyDialog + EmpatheticDialogues listener turns) plus synthetic
memory-grounded examples (recall, 'I don't know yet', and replies that use a remembered fact).
Values (names, cities, jobs...) are split into train / held-out pools so copying can be tested on unseen values."""
import random
from ovi import dialogue as D

NAMES = "Max Bella Luna Charlie Rocky Daisy Milo Lucy Cooper Sadie Buddy Molly Bailey Maggie Toby Coco Oscar Ruby Duke Lola Zeus Penny Jack Rosie Teddy Nala Leo Willow Bear Ginger Finn Pepper Sam Olive Murphy Hazel Gus Piper Otis Stella Moose Roxy Ziggy Maple Archie Nova Biscuit Peanut Waffles Pickles Clover Rusty Sunny Tucker Winnie Banjo Mochi Pixel Tigger Shadow Smokey".split()
CITIES = "Oslo Lisbon Denver Austin Madrid Toronto Dublin Boston Seattle Chicago Berlin Vienna Prague Lyon Perth Cairo Mumbai Pune Delhi Lahore Dhaka Nairobi Lagos Accra Quito Lima Bogota Helsinki Bergen Leeds Bristol Glasgow Phoenix Portland Orlando Tampa Miami Atlanta Dallas Houston Sydney Auckland Osaka Kyoto Seoul Busan Taipei Hanoi Manila Jakarta".split()
JOBS = "nurse teacher plumber chef doctor engineer lawyer farmer pilot designer writer mechanic dentist electrician accountant baker driver architect pharmacist carpenter cashier librarian programmer photographer firefighter".split()
FOODS = "pizza sushi tacos pasta curry ramen burgers salad dumplings lasagna pancakes noodles biryani falafel steak waffles risotto burritos".split()
HOBBIES = "hiking painting chess cycling knitting gardening swimming baking fishing singing climbing photography cooking running reading dancing sailing skating surfing".split()
COLORS = "blue green red purple yellow orange black white pink teal".split()
PEOPLE = "Anna Ben Carlos Dina Eli Farah Greg Hana Ian Jonas Kara Liam Mia Noor Omar Pia Quinn Rita Sven Tara Uma Vik Wes Yara Zane".split()
_rs = random.Random(7); _pools = {}
for _n, _l in [("names", NAMES), ("cities", CITIES), ("jobs", JOBS), ("foods", FOODS), ("hobbies", HOBBIES), ("colors", COLORS), ("people", PEOPLE)]:
    l = _l[:]; _rs.shuffle(l); k = max(2, len(l) // 5); _pools[_n] = (l[k:], l[:k])      # (train, heldout)


def pick(rnd, pool, held):
    return rnd.choice(_pools[pool][1 if held else 0])


def an(w):
    return ("an " if w[0].lower() in "aeiou" else "a ") + w


def cap(s):
    return s[0].upper() + s[1:]


def kinds(rnd, held):
    """Each fact: (memory line, [recall questions], [recall answers], slot name)."""
    v = lambda p: pick(rnd, p, held)
    d, c, j, f, h, col, p = v("names"), v("cities"), v("jobs"), v("foods"), v("hobbies"), v("colors"), v("people")
    return {
        "dog": (f"your dog is named {d}", ["what's my dog's name?", "what is my dog called?", "do you remember my dog's name?", "what was my dog's name again?", "can you tell me my dog's name?"],
                [f"Your dog's name is {d}.", f"You told me your dog is called {d}.", f"It's {d}, your dog.", f"Your dog is named {d}."], d),
        "city": (f"you live in {c}", ["where do I live?", "which city do I live in?", "do you know where I live?", "what city am I in?", "where am I from again?"],
                 [f"You live in {c}.", f"You told me you live in {c}.", f"You're in {c}.", f"It's {c}, right?"], c),
        "job": (f"you work as {an(j)}", ["what do I do for a living?", "what's my job?", "what do I do for work?", "do you remember what my job is?", "what is my profession?"],
                [f"You work as {an(j)}.", f"You're {an(j)}.", f"You told me you work as {an(j)}."], j),
        "food": (f"your favorite food is {f}", ["what's my favorite food?", "what do I like to eat the most?", "do you know my favorite food?", "what food do I love?"],
                 [f"Your favorite food is {f}.", f"You love {f}.", f"You told me {f} is your favorite."], f),
        "hobby": (f"you enjoy {h}", ["what do I like to do for fun?", "what's my hobby?", "what do I do in my free time?", "do you remember my hobby?"],
                  [f"You enjoy {h}.", f"You told me you like {h}.", f"{cap(h)} is your hobby."], h),
        "color": (f"your favorite color is {col}", ["what's my favorite color?", "which color do I like best?", "do you remember my favorite color?"],
                  [f"Your favorite color is {col}.", f"You like {col} best.", f"It's {col}."], col),
        "friend": (f"your friend is named {p}", ["who is my friend?", "what's my friend's name?", "do you remember my friend's name?", "what was my friend called?"],
                   [f"Your friend's name is {p}.", f"You told me your friend is {p}.", f"It's {p}."], p),
    }


NO_MEM = ["I don't think you've told me that yet. What is it?", "I don't know that yet. Can you tell me?", "You haven't told me that. What is it?",
          "I'm not sure, you haven't mentioned it. What is it?", "I don't remember you saying. Tell me?"]
USES = {   # (fact kind) -> [(user statement, reply with {v})]
    "dog": [("my dog is sick", ["Oh no, I'm sorry. I hope {v} feels better soon.", "That's sad. Is {v} going to see the vet?", "I'm sorry to hear that. I hope {v} gets well soon."]),
            ("my dog barked all night", ["Oh dear. Was {v} upset about something?", "That sounds tiring. Is {v} usually that loud?"]),
            ("I just took my dog for a walk", ["Nice! Did {v} enjoy it?", "That's good. I bet {v} loved it."]),
            ("I'm so happy today", ["That's great to hear! Did you do something fun with {v}?", "I'm glad! Is {v} part of the good mood?"])],
    "city": [("it's so hot today", ["It must be quite warm in {v} then. Stay cool!", "Hot days in {v} sound tiring. Drink some water."]),
             ("I'm bored", ["Is there something fun to do in {v}?", "Maybe a walk around {v} would help?"]),
             ("the traffic was terrible", ["Traffic in {v} can be rough. I'm sorry.", "Ugh, that sounds stressful. Is it always like that in {v}?"]),
             ("what should I do this weekend?", ["Maybe explore a part of {v} you haven't seen yet?", "How about visiting a park in {v}?"])],
    "job": [("work was awful today", ["I'm sorry. Being {v} can be draining. Do you want to talk about it?", "That sounds rough. Was it a hard day as {v}?"]),
            ("I'm so tired", ["Long days as {v} will do that. Try to rest tonight.", "Being {v} is tiring. Get some sleep if you can."]),
            ("I got a raise", ["Congratulations! You deserve it as {v}.", "That's wonderful news! Hard work as {v} paid off."])],
    "food": [("I'm hungry", ["How about some {v}? It's your favorite.", "Maybe {v}? You love it."]),
             ("what should I cook tonight?", ["Since you love {v}, how about that?", "You could make {v}. It's your favorite."])],
    "hobby": [("I have free time today", ["It sounds like a good day for {v}.", "Why not do some {v}? You enjoy it."]),
              ("I'm stressed", ["Maybe some {v} would help you relax.", "Try {v}. You enjoy it, right?"])],
    "friend": [("I'm feeling lonely", ["Maybe call {v}? I bet they would love to hear from you.", "Have you talked to {v} lately?"]),
               ("I had a fight with my friend", ["I'm sorry. Is it {v}? I hope you two make up soon.", "That's hard. Do you want to talk it through, {v} matters to you."])],
}
OTHER_USER = [("hello", "Hi! How are you doing today?"), ("how are you", "I'm doing well, thanks for asking. How about you?")]


def synth(n, seed, held=False):
    rnd = random.Random(seed); out = []
    for _ in range(n):
        ks = kinds(rnd, held); keys = list(ks)
        pool = rnd.sample(keys, rnd.randint(1, 3)); k = pool[0]
        mem = [ks[x][0] for x in pool]; rnd.shuffle(mem)
        r = rnd.random()
        if r < 0.45:                                           # recall from memory
            _, qs, ans, val = ks[k]
            out.append(([("u", rnd.choice(qs))], rnd.choice(ans), mem, val))
        elif r < 0.60:                                         # asked about something never told
            missing = rnd.choice([x for x in keys if x not in pool])
            out.append(([("u", rnd.choice(ks[missing][1]))], rnd.choice(NO_MEM), mem, None))
        else:                                                  # use a remembered fact in ordinary talk
            kk = rnd.choice([x for x in pool if x in USES] or [None])
            if kk is None:
                u, a = rnd.choice(OTHER_USER); out.append(([("u", u)], a, mem, None)); continue
            u, tpls = rnd.choice(USES[kk]); val = ks[kk][3]
            out.append(([("u", u)], rnd.choice(tpls).format(v=val), mem, val))
    return out


def real(split_a, split_b):
    h = D.triples(split_a, "daily") + D.triples(split_b, "emp")
    return [([("u" if (len(hh) - i) % 2 == 1 else "o", t) for i, t in enumerate(hh)], r, [], None) for hh, r in h]
