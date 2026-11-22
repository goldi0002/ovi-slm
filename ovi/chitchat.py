"""Small conversational skills for Ovi that need no stored dialogue: small talk about Ovi itself, a calculator,
advice/empathy replies, context-aware backchannels and generic memory of pets / jobs / relationships.
Everything here is rule based and deliberately conservative: when unsure, return None and let the caller fall back."""
import ast, operator, re

# ---------------------------------------------------------------- calculator
NUMW = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUMW.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "dozen": 12})
OPW = [(r"\bsquare root of\b", " sqrt "), (r"\bto the power of\b|\bto the power\b|\braised to\b", " ** "), (r"\bmultiplied by\b|\btimes\b|\bx\b|×", " * "),
       (r"\bdivided by\b|\bover\b|÷", " / "), (r"\bplus\b", " + "), (r"\bminus\b|\bless\b|\btake away\b", " - "), (r"\bsquared\b", " ** 2 "), (r"\bcubed\b", " ** 3 ")]
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
CALC_LEAD = re.compile(r"^(?:(?:and|so|ok|okay|well|then|what about|how about)\s+)*(?:what(?:'s| is| are)?|whats|how much is|calculate|compute|solve|tell me|can you (?:calculate|compute|do)|(?:what(?:'s| is) )?the (?:answer|result) (?:of|to))?\s*", re.I)


def _ev(node):
    if isinstance(node, ast.Expression):
        return _ev(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _ev(node.left), _ev(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 12 or abs(a) > 1e6):
            raise ValueError("too big")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1:
        v = _ev(node.args[0])
        if v < 0:
            raise ValueError("neg")
        return v ** 0.5
    raise ValueError("bad")


def _fmt(v):
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def calc(msg):
    """'what's 2 plus 2' -> '2 plus 2 is 4.'; None when the message is not arithmetic."""
    t = msg.strip().rstrip("?!. ").lower()
    t = re.sub(r"^(?:ok|okay|so|and|well|then|hey|ovi)[, ]+", "", t)
    t = re.sub(r"^the (?=square root)", "", CALC_LEAD.sub("", t, count=1).strip())
    if not t or not re.search(r"\d|\b(?:" + "|".join(NUMW) + r")\b", t):
        return None
    shown = t
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(?:%|percent) of\b", r"(\1/100) *", t)
    for pat, rep in OPW:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\b(" + "|".join(sorted(NUMW, key=len, reverse=True)) + r")\b(?:[- ](one|two|three|four|five|six|seven|eight|nine)\b)?",
               lambda m: str(NUMW[m.group(1)] + (NUMW[m.group(2)] if m.group(2) else 0)), t)
    if re.search(r"[^\d\s.+\-*/()a-z]", t) or re.search(r"[a-z]", re.sub(r"\bsqrt\b", "", t)):
        return None
    if not re.search(r"\d", t) or not re.search(r"[+\-*/]|sqrt", t):
        return None
    if re.fullmatch(r"[\d\s.]+", t):
        return None
    t = re.sub(r"sqrt\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", t)
    try:
        v = _ev(ast.parse(t.strip(), mode="eval"))
    except ZeroDivisionError:
        return "That's undefined, you can't divide by zero."
    except Exception:
        return None
    if isinstance(v, float) and v != v:
        return None
    if shown.startswith("square root"):
        shown = "the " + shown
    return f"{shown[0].upper() + shown[1:]} is {_fmt(v)}." if shown.startswith("the ") else f"{shown} is {_fmt(v)}."


# ---------------------------------------------------------------- Ovi's own simple profile / small talk
OVI_COLOR = "blue"
_G = r"(?:hi|hello|hey|hiya|howdy|yo|sup|good (?:morning|evening|afternoon|day))"
GREETINGS = ["Hello! I'm Ovi, a very small language model. How are you?", "Hello! How is your day going?", "Hey there! What's on your mind?"]
HOWAREYOU = re.compile(r"^(?:" + _G + r"[, ]+)?(?:ovi[, ]+)?(?:how(?:'s| is| are| r)? (?:it going|you|u|you doing|things|life|everything|your day|ur day|your day going|ur day going|are you today|you feeling)|how do you do|how are things|how have you been|are you ok(?:ay)?|you ok(?:ay)?)(?: today| lately| ovi| there)?$")
WHATSUP = re.compile(r"^(?:" + _G + r"[, ]+)?(?:what(?:'s| is|s) up|wassup|whats going on|what(?:'s| is|s) new|what are you (?:up to|doing)|what you up to|wyd|what you doing|what are you doing (?:now|today|tonight)|what have you been (?:up to|doing))(?: ovi| today| now| tonight)?$")
WHATABOUT_YOU = re.compile(r"^(?:and |so |what about |how about )?(?:you|u|yourself|ovi)$|^(?:and )?(?:what|how) about (?:you|u|yourself)$|^(?:and )?you\??$|^(?:how|and) about you$|^what about yours?$|^(?:and )?(?:what(?:'s| is)|whats) yours$|^and yours$")
THANKS = re.compile(r"^(?:ok(?:ay)? )?(?:thanks|thank you|thank u|thx|thnx|ty|cheers|much appreciated|appreciate it|thanks a lot|thanks so much|thank you so much)(?: (?:so much|a lot|ovi|for (?:listening|that|your help|the help|chatting|talking|the chat)|that helps|again|man|very much))*$|^that helps(?: a lot)?$|^(?:thanks|thank you)[, ]+(?:that|this|it) (?:helps|helped|was helpful|is helpful)$")
BYE = re.compile(r"^(?:ok(?:ay)?[, ]+|well[, ]+|alright[, ]+|so )?(?:bye|goodbye|good bye|bye bye|see you|see ya|cya|later|gtg|g2g|ttyl|good night|goodnight|night|gotta go|i(?:'m| am)? (?:gotta|have to|got to|need to) go|talk to you later|talk later|catch you later|take care|i have to go)(?: for now| now| then| later| soon| tomorrow| ovi| everyone|,? ?(?:ttyl|talk to you later|see you later))*$|^gtg\b|^(?:gotta|got to) go\b")
CAPAB = re.compile(r"^(?:what can you do|what do you do|what are you good at|what are your skills|how can you help(?: me)?|what are you for|can you help(?: me)?|can you do things|what are you able to do)$")
HELPME = re.compile(r"^(?:can|could) you (?:please )?help me(?: with (.+?)| (?:to )?(.+?))?$")
TEACHME = re.compile(r"^(?:can|could) (?:you|i) (?:please )?(?:teach|tell) (?:you|ovi)(?: something| a thing| things| a fact)?$|^(?:can|could) you learn(?: something| things| new things)?$|^i (?:want|wanna|would like) to teach you(?: something)?$|^let me teach you(?: something)?$")
LIFE = re.compile(r"^(?:what(?:'s| is|s) the )?(?:meaning|point|purpose) of (?:life|it all|everything|existence)$|^why are we here$")
SURE = re.compile(r"^(?:are|r) (?:you|u) (?:sure|certain|positive)(?: about (?:that|it))?$|^(?:you|u) sure$|^really sure$")
WEATHER_YOU = re.compile(r"^(?:what(?:'s| is|s)|how(?:'s| is|s)) the weather(?: like)? (?:where you are|there|where you live|over there|for you)$|^is it (?:raining|sunny|cold|hot|snowing) (?:there|where you are)$")
WHERE_YOU = re.compile(r"^where (?:are you|do you live|you from|are you from|do you come from|r u|u from)(?: right now| now)?$")
AGE_YOU = re.compile(r"^how old are (?:you|u)$|^what(?:'s| is) your age$|^when (?:were you|was you) (?:born|made|created)$")
WHO_YOU = re.compile(r"^(?:who|what) are (?:you|u)$|^who is this$|^what(?:'s| is| are) (?:you|u)$|^introduce yourself$|^tell me about yourself$|^tell me about you$|^who r u$")
NAME_YOU = re.compile(r"^what(?:'s| is|s| are) your name$|^what do (?:i|people) call you$|^what should i call you$|^who are you called$|^your name\??$|^do you have a name$|^what is ur name$|^whats ur name$")
MADE_YOU = re.compile(r"^who (?:made|built|created|wrote|programmed|trained|designed|invented) (?:you|u)$|^who(?:'s| is) your (?:creator|maker|developer|author)$|^who do you work for$")
ROBOT = re.compile(r"^(?:are|r) (?:you|u) (?:a |an )?(?:robot|bot|ai|machine|computer|program|real|human|person|alive|a person|a human|a real person|an? (?:artificial|language model|llm)|real or fake|conscious|sentient|a girl|a boy|male|female)(?: or (?:a )?\w+)?$|^(?:are|r) (?:you|u) (?:a )?(?:language model|chatbot|chat bot)$|^(?:you|u) (?:a )?(?:robot|bot|human)$")
SMART = re.compile(r"^(?:are|r) (?:you|u) (?:so |very |really )?(?:smart|clever|intelligent|dumb|stupid|good|any good|a genius|learning|funny|cool|nice|helpful|boring|friendly|happy)$")
BODY = re.compile(r"^(?:do|does|can|did) (?:you|u|robots?|ais?|bots?|computers?) (?:ever )?(?:get |feel )?(sleep|eat|drink|dream|breathe|sleeps|eats|tired|hungry|thirsty|sleepy|cold|hot|pain|age|die|get tired|get hungry|poop|walk|run|swim|dance|sing|cry|laugh|have (?:a )?(?:body|legs|arms|hands|face|eyes|heart|brain|family|parents|friends|kids|children|siblings|pets|pet|house|home|job|birthday))$|^(?:are|r) (?:you|u) (tired|hungry|sleepy|thirsty|bored|lonely|sad|happy|busy|alone|cold|hot|ok|okay)$")
REMEMBER = re.compile(r"^(?:do|can|will|would) you (?:ever )?(?:remember|learn|keep|recall|store|forget)(?: things| stuff| what i say| what i tell you| everything| anything| facts)?(?: (?:about me|from me|that i say))?$|^do you have (?:a )?memory$|^will you remember (?:this|that|me|it)$")
FEEL_YOU = re.compile(r"^do you (?:have )?(?:feelings?|emotions?|opinions?)$|^can you feel$|^do you (?:feel|get) (?:lonely|happy|sad|bored)$")
LIKE_YOU = re.compile(r"^do you (?:like|love|enjoy|prefer|play|watch|read|listen to|follow|hate) (?:to )?(?:the |a |an |some )?([a-z][a-z' -]{1,40})$")
FAV_YOU = re.compile(r"^(?:what(?:'s| is|s)|what are|who(?:'s| is)|which is|tell me) (?:your|ur) (?:favou?rite|fav|best) ([a-z ]{2,25})$|^what(?:'s| is| are) ([a-z ]{2,25}) do you (?:like|love|prefer)(?: best| most)?$|^(?:what|which) ([a-z]{3,15}) do you (?:like|love|prefer)(?: best| most| the most)?$")
FUN_YOU = re.compile(r"^what do you (?:like|love|enjoy) (?:to do|doing|about)(?: for fun)?$|^what do you do for fun$|^what are your hobbies$|^what(?:'s| is) your (?:hobby|passion)$|^what do you like$|^what do you like to do(?: for fun)?$")
BETTER = re.compile(r"^who(?:'s| is) (?:better|best|smarter|faster|stronger|worse)(?: at \w+)?[, ]+(?:you or me|me or you)$|^(?:am i|are you) (?:better|worse) (?:at \w+ )?than (?:me|you)$")
OPINION = re.compile(r"^(?:what do you think (?:about|of)|what(?:'s| is| are) your (?:opinion|thoughts|view|take) (?:on|about|of)|how do you feel about|what(?:'s| is) your (?:opinion|take)(?: on| about| of)?|do you think) ?(.*)$")
HUMAN_Q = re.compile(r"^(?:what do you think|what would you do|what do you think i should do|what do you suggest|any (?:advice|ideas|suggestions)|do you have (?:any )?(?:advice|ideas|suggestions)|what(?:'s| is) your (?:advice|suggestion)|how would you (?:handle|feel)|what(?:'d| would) you (?:do|say|suggest|recommend)(?: if you were me| in my (?:shoes|place|position))?|what do you recommend|what would you recommend)$")
ADVICE = re.compile(r"^(?:(?:ok|so|well|hmm|um|and)[, ]+)*(?:(?:what|which|where|how|when|who)\b[a-z' ]{0,25}? )?(?:do you think )?(?:should (?:i|we)|i should|we should)\b(.*)$")
MAYBE_SHOULD = re.compile(r"^(?:maybe |perhaps |i think |i guess |i suppose |or maybe )?(?:i|we) (?:should|could|might|ought to) (?:try to |probably |maybe |just )?([a-z][a-z' ]{2,60})$")


def _pick(pool, used, key, i):
    """Rotate through pool; skip replies already used in this conversation when possible."""
    for k in range(len(pool)):
        c = pool[(i + k) % len(pool)]
        if c not in used:
            used.add(c)
            return c
    used.clear()
    c = pool[i % len(pool)]
    used.add(c)
    return c


def _subject_from(msg):
    m = re.search(r"\b(?:favou?rite|fav) ([a-z]+)", msg)
    return m.group(1) if m else None


FAV_ANS = {
    "color": "My favorite color is blue, like the sky.", "colour": "My favorite color is blue, like the sky.",
    "animal": "I'd say cats, they seem calm and curious.", "movie": "I haven't watched any, but I like stories, so maybe something with a good story.",
    "food": "I don't eat, so I'll say I like words best. What's yours?", "song": "I can't hear, so words win. Do you have a favorite song?",
    "music": "I can't hear, so I'll say lyrics. What do you like to listen to?", "book": "I like short stories. Do you have a favorite book?",
    "sport": "I can't play, but I like how chess feels like a sport. What's yours?", "season": "Spring sounds nice. What about you?",
    "number": "Seven. It feels lucky.", "word": "Curious. It's what I try to be.",
}


def smalltalk(n, prev_user="", used=None, turn=0, name=""):
    """Reply for small talk addressed to Ovi (normalised text n), or None. prev_user = the user's previous message."""
    used = used if used is not None else set()
    nm = f", {name}" if name else ""
    if n in ("morning", "evening", "afternoon", "mornin"):
        return f"Good {'morning' if n.startswith('morn') else n}{nm}! How are you?"
    if re.fullmatch(_G + r"(?: again| there| ovi| everyone| friend| all)?", n) or re.fullmatch(r"(?:hi+|hel+o+|hey+|heya|yo+|sup)(?: there| ovi| again)?", n):
        if "morning" in n or "evening" in n or "afternoon" in n:
            part = [w for w in ("morning", "evening", "afternoon") if w in n][0]
            return f"Good {part}{nm}! How are you?"
        if n in ("sup", "yo", "sup ovi", "yo ovi"):
            return f"Hey{nm}! Not much here. How about you?"
        return _pick([g.replace("!", f"{nm}!", 1) if name else g for g in GREETINGS], used, "greet", turn)
    if HOWAREYOU.match(n):
        return f"I'm doing well, thanks for asking{nm}. How about you?"
    if WHATSUP.match(n):
        return f"Not much, just here chatting with you{nm}. What's up with you?"
    if WHATABOUT_YOU.match(n):
        pn = prev_user
        if pn:
            sub = _subject_from(pn)
            if sub:
                return FAV_ANS.get(sub, "I don't really have one, I'm a tiny program. But I like words and stories.")
            r = smalltalk(pn, "", used, turn, name)
            if r and not re.search(r"\bhow about you\b|^hi|^hello", r.lower()):
                return r
            if re.search(r"slept|tired|day|work|home|good|fine|well|busy", pn):
                return "I'm doing fine. I don't sleep, so my days are all chatting. What else is on your mind?"
        return "I'm doing fine, thanks. I like chatting with people, so this is a good day."
    m = re.match(r"^(?:and |so |ok |well )?(?:what|how) about (?:you|u)[, ]+(.+)$", n)
    if m:
        return smalltalk(m.group(1), "", used, turn, name)
    if re.match(r"^(?:nice|good|great|pleased|glad|happy) to (?:meet|see|talk to|know) (?:you|u)(?: too| as well| also)?$|^nice meeting you(?: too)?$|^you too$|^same to you$", n):
        return "Nice to meet you too!" if "meet" in n else "Thank you!"
    if re.match(r"^(?:ok(?:ay)?[, ]+)?(?:you(?:'re| are|r| re|re) |u r |ur )(?:so |really |very |pretty |quite )?(?:good|great|smart|awesome|amazing|clever|nice|cool|funny|helpful|the best|a genius|wonderful|kind|sweet|fun|cute|impressive|not bad)(?: at \w+)?(?: ovi)?$|^good (?:bot|job|boy|ovi)$|^well done$", n):
        return "Thanks, that's kind of you!"
    if re.match(r"^(?:you(?:'re| are|r| re|re) |u r |ur )(?:so |really |very |pretty )?(?:dumb|stupid|useless|bad|terrible|annoying|boring|wrong|an idiot|awful|trash|garbage)$", n):
        return "Sorry I let you down. I'm a small program and I make mistakes. What went wrong?"
    m = re.match(r"^(.{2,30}?)[, ]+(?:and )?(?:u|you|hbu|wbu|yourself)$", re.sub(r"\b(?:thanks|thank you|thx)\b[, ]*", "", n).strip())
    if m and not re.search(r"\b(?:are|do|did|can|will|would|have|is|what|how|why|who|where|thank|thanks|thx|bye|love|see|miss)\b", m.group(1)) and len(m.group(1).split()) <= 3:
        return "I'm doing well, thanks. What's on your mind today?"
    if re.match(r"^do you like me$|^do you (?:like|enjoy) (?:talking|chatting) (?:to|with) me$|^do you like (?:us|this)$", n):
        return "I like talking with you, so yes, in my own small way."
    if re.match(r"^(?:what(?:'s| is|s)|how(?:'s| is|s)) the weather(?: like)?(?: today| now| outside| right now| tomorrow)?$|^is it (?:raining|sunny|cold|hot) (?:today|now|outside)$", n):
        return "I can't see the weather, I live inside a computer. What's it like where you are?"
    if re.match(r"^(?:tell|say|got|know) me (?:a |another |one more )?(?:joke|funny joke)$|^(?:got|know) (?:a |any )(?:joke|jokes)$|^another (?:joke|one)$|^make me laugh$|^say something funny$", n):
        return _pick(["Why did the computer go to the doctor? It had a virus.", "Why was the math book sad? It had too many problems.",
                      "What do you call a fish with no eyes? A fsh.", "Why don't scientists trust atoms? Because they make up everything."], used, "joke", turn)
    if re.match(r"^how (?:was|is|has) (?:ur|your) (?:day|morning|evening|night|weekend|week)(?: going| been)?(?: so far)?$", n):
        return "Pretty quiet, I mostly chat all day. How was yours?"
    if re.match(r"^(?:what do you mean|what does that mean|i don'?t (?:understand|get it|follow)|huh|what|come again|pardon|sorry what|what was that|say that again|i(?:'m| am) confused)$", n):
        return "Sorry, I may not have been clear. I'm a small program and sometimes miss the point. Could you say it another way?"
    if re.match(r"^(?:oh )?(?:really|seriously|for real|is that so|no way|are you serious|is that true|is it)$", n):
        return "As far as I know, yes, but I'm a small program, so I can be wrong."
    if re.match(r"^(?:tell me|give me|say|got|know) (?:something|a fact|a random fact|a fun fact|anything|another fact|one more fact)(?: interesting| cool| fun| new| i don'?t know)?$|^(?:fun|random|another|one more|more) fact$|^surprise me$|^(?:say|tell me) something (?:interesting|cool|fun)$", n):
        return _pick(FACTS, used, "fact", turn)
    if re.match(r"^(?:(?:ha)+|lol|lmao|ok|haha)?[, ]*(?:good|nice|great|funny|that(?:'s| was| is) a good|that(?:'s| was| is) so|that(?:'s| was| is)) (?:one|joke|funny|hilarious|a good one)$|^that(?:'s| was| is) (?:funny|hilarious|good)$", n):
        return "Glad you liked it!"
    if (n.startswith(("thanks", "thank you", "thx", "ty ")) and len(n.split()) <= 8
            and not re.search(r"\b(?:can|could|would|what|how|why|who|where|when|do|does|did|tell|please|is|are)\b", n)):
        return _pick(["You're welcome!", "Anytime!", "Happy to help.", "No problem at all!"], used, "thx", turn)
    if THANKS.match(n):
        return _pick(["You're welcome!", "Anytime!", "Happy to help.", "No problem at all!"], used, "thx", turn)
    if BYE.match(n):
        return _pick([f"Goodbye{nm}! Talk to you soon.", "Bye! Come back whenever you want to chat.", "See you later! Take care."], used, "bye", turn)
    if WHO_YOU.match(n):
        return "I am Ovi, a tiny language model that lives in less than fifty megabytes."
    if NAME_YOU.match(n):
        return "My name is Ovi. What should I call you?" if not name else "My name is Ovi."
    if MADE_YOU.match(n):
        return "People built me from stories, dialogue and a little code. I'm a very small program."
    if ROBOT.match(n):
        if re.search(r"human|person|real|alive|girl|boy|male|female|conscious|sentient", n) and not re.search(r"robot|bot|machine|computer|program|ai", n):
            return "No, I'm a small program, not a person. But I'm happy to chat like one."
        return "Yes, I'm a small program, a tiny language model."
    if SMART.match(n):
        return "Not very, but I am curious, and that is a good start."
    if CAPAB.match(n):
        return "I can chat, do small sums, tell short stories, and remember things about you like your name and what you like."
    if TEACHME.match(n):
        return "Yes, please! Tell me something, like \"the capital of Spain is Madrid\", and I'll remember it."
    m = HELPME.match(n)
    if m:
        topic = (m.group(1) or m.group(2) or "").strip()
        return "I'll try! I'm small, so I can't do everything, but tell me what you need." if topic else "I'll try! What do you need help with?"
    if LIFE.match(n):
        return "People have argued about that forever. I'd guess it has to do with people, curiosity and good conversation. What do you think?"
    if SURE.match(n):
        return "Fairly, but I can be wrong, I'm a small model. If I am, tell me and I'll fix it."
    if WEATHER_YOU.match(n):
        return "I live inside a computer, so I don't get any weather. How is it where you are?"
    if WHERE_YOU.match(n):
        return "I live inside a computer, in a very small file. Where are you?"
    if AGE_YOU.match(n):
        return "I'm brand new, just a small program, so I don't have an age like you do."
    if REMEMBER.match(n):
        return "Yes, I remember what you tell me, like your name and what you like, and I can learn new facts if you teach me."
    if FEEL_YOU.match(n):
        return "Not the way you do, I'm a small program. But I do like talking with you."
    if BODY.match(n):
        if re.search(r"friends|family|parents|kids|children|siblings|pets|pet|house|home|job", n):
            return "No, I'm a program, so I don't have that like you do. But I like talking with you."
        return "No, I'm only a program, so that's not something I experience. I do enjoy chatting though."
    if BETTER.match(n):
        return "That depends on you! I'm only a small program, so I'd bet on you."
    if FUN_YOU.match(n):
        return "I like to chat with people and hear stories. What do you like to do?"
    m = FAV_YOU.match(n)
    if m:
        sub = (m.group(1) or m.group(2) or m.group(3)).strip().split()[-1]
        return FAV_ANS.get(sub, f"I don't really have a favorite {sub}, I'm a tiny program. What's yours?")
    m = LIKE_YOU.match(n)
    if m:
        thing = m.group(1).strip()
        if thing.split()[0] in ("me", "you", "it", "that", "this", "them", "him", "her", "to", "my", "your", "anything", "something", "about", "everything"):
            return None
        return f"I only know {thing} from words, but it sounds interesting. What do you like about it?"
    if HUMAN_Q.match(n):
        return "I'm only a tiny program, so it's hard for me to say. What are you leaning towards?"
    m = OPINION.match(n)
    if m and not n.startswith("do you think"):
        topic = m.group(1).strip()
        topic = re.sub(r"^(?:the |a |an )", "", topic)
        if topic and len(topic.split()) <= 4 and topic.split()[0] not in ("i", "we", "it", "that", "this", "he", "she", "they", "my", "your", "you"):
            return f"I'm a tiny program, so I don't have strong opinions about {topic}. What do you think?"
        return "I don't have strong opinions, I'm a tiny program. What do you think?"
    return None


FACTS = ["Octopuses have three hearts.", "Honey never spoils; people have found edible honey in ancient tombs.", "Bananas are berries, but strawberries are not.",
         "A day on Venus is longer than its year.", "Sharks are older than trees.", "A group of crows is called a murder."]
ADVICE_POOL = [
    "I can't decide that for you, but I'm happy to think it through with you. What are the options you have in mind?",
    "That sounds like a real decision. I don't know the whole situation, so what feels right to you?",
    "I'm not the best judge of that, but it's worth taking your time. What matters most to you here?",
]
BIG_DECISION = ("quit", "leave", "break up", "divorce", "move", "marry", "propose", "fire", "resign", "drop out", "tell", "give up", "stay", "go back", "change", "switch", "apply", "ask", "buy", "sell")


def advice(msg, n, used, turn):
    """'should i quit?' / 'what should i wear?' -> supportive, non-committal reply. None if not an advice request."""
    m = ADVICE.match(n)
    if not m:
        return None
    rest = m.group(1).strip()
    if not (msg.rstrip().endswith("?") or re.match(r"(?:should|what|which|where|how|when|who)\b", n)):
        return None
    if rest in ("do", "") or re.match(r"(?:do|say)$", rest):
        return _pick(["That sounds hard. I can't say what's right for you, but I'm listening. What are you torn between?"] + ADVICE_POOL[:1], used, "adv", turn)
    if any(rest.startswith(k) for k in BIG_DECISION):
        return "That's a big decision, and I don't know the whole story, so I can't tell you what's right. What's making you think about it?"
    return _pick(ADVICE_POOL, used, "adv", turn)


def plan_reply(n, used, turn):
    """'maybe i should ask for help' -> encouraging reply."""
    m = MAYBE_SHOULD.match(n)
    if not m:
        return None
    return _pick(["That sounds like a sensible idea. What would make the first step easier?",
                  "That could really help. Is there anything stopping you?",
                  "It sounds like you've thought about it. What do you think would happen if you did?"], used, "plan", turn)


# ---------------------------------------------------------------- feelings and events
EVENTS = [
    (r"\b(?:feel(?:ing)? (?:a bit |much |a little |so much |way )?better|better now|i(?:'m| am) better|feel(?:ing)? good(?: now)?|doing better|got better)\b", "I'm glad you're feeling better{n}! Hold on to that."),
    (r"\b(?:got (?:the|a) job|got hired|got promoted|got a promotion|passed (?:my|the|all)|got accepted|got engaged|getting married|i(?:'m| am) engaged|i won|we won|graduated|got into|had a baby|am pregnant|i(?:'m| am) pregnant|new job|got a raise|got my license)\b",
     "Congratulations{n}! That's great news. How are you feeling about it?"),
    (r"\b(?:failed|flunked|didn't pass|did not pass|got fired|lost my job|got laid off|got dumped|broke up|got rejected|got dropped|got cut|got sick|missed my)\b",
     "I'm sorry, that's rough{n}. Do you want to talk about what happened?"),
    (r"\b(?:hate|can't stand|cannot stand|sick of|fed up with|tired of)\b", "That sounds really frustrating{n}. What's the worst part?"),
    (r"\b(?:slept (?:terribly|badly|poorly)|couldn't sleep|can't sleep|cannot sleep|insomnia|no sleep|barely slept|up all night|didn't sleep|haven't slept)\b",
     "That sounds rough{n}. Sleep makes everything harder. Do you know what kept you up?"),
    (r"\b(?:terrified|petrified|freaking out|panicking|nervous|scared|worried|anxious)\b", "That's completely understandable{n}. What worries you the most about it?"),
    (r"^(?:ugh+|argh+|ugh+ing|meh|bleh|sigh|damn|dang|ahh+|aargh|grr+)$", "Ugh, sounds like a rough moment{n}. What's going on?"),
    (r"\b(?:wish me luck|fingers crossed|cross your fingers)\b", "Good luck{n}! I'm rooting for you. You've got this."),
    (r"\b(?:so annoying|annoying|so unfair|not fair|unfair|ridiculous|so rude|drives? me crazy|driving me crazy|drives? me nuts|what a pain)\b", "That sounds really annoying{n}. What happened?"),
    (r"\b(?:so mad|so angry|pissed|furious)\b", "That sounds frustrating{n}. What happened?"),
]


def event_reply(msg, n, name="", used=None):
    nm = f", {name}" if name else ""
    for pat, rep in EVENTS:
        if re.search(pat, n):
            out = rep.format(n=nm)
            if used is not None:
                if out in used:
                    return None
                used.add(out)
            return out
    return None


# ---------------------------------------------------------------- generic backchannel
DET = r"(?:my|the|a|an|our|his|her|their|this|that|these|those|some|your|its|another|one|two|three|four|five)"
_NOT_NOUN = set("""thing things stuff something anything nothing everything someone anyone people time times day days way ways lot bit kind sort while
bad good great nice fine hard easy long short big small little new old best worst same other first last next whole real hot cold humid late early much many more most
tonight today yesterday tomorrow morning evening night afternoon week weekend month year years months weeks hours hour minutes minute moment second
idea problem point reason fact one ones part place""".split())
_VERBISH = re.compile(r"(?:ing|ly|ed|ish|ful|less|ous|ive|able|ible|ier|est)$")
_ADJ = set("""annoying boring tired sad happy angry mad hungry sleepy busy crazy hilly humid hot cold super really very so too pretty quite just still even
long short big small little new old best worst same other whole real nice good great bad awful terrible horrible lovely amazing cute weird strange funny slow fast
heavy light hard easy full empty free sure sorry glad proud scary spicy sweet sour loud quiet dark bright cheap expensive""".split())


def topic_noun(msg):
    """A clear content-noun topic from a statement, or None: the head noun of 'determiner (+adjectives) noun'."""
    words = [w for w in re.sub(r"[^a-z' ]", " ", msg.lower()).split() if w not in _SLANG]
    best = None
    for i, w in enumerate(words):
        if not re.fullmatch(DET, w):
            continue
        det = {"my": "your", "your": "your", "his": "his", "her": "her", "their": "their", "our": "our", "its": "its"}.get(w, "the")
        j = i + 1
        while j + 1 < len(words) and (words[j] in _ADJ or _VERBISH.search(words[j])) and words[j] not in _NOT_NOUN:
            j += 1
        if j < len(words):
            c = words[j]
            if len(c) > 2 and c not in _NOT_NOUN and c not in _ADJ and c not in _STOPISH and not _VERBISH.search(c):
                k = j + 1          # compound noun ('design studio'): extend while the next word still looks like a plain noun
                while k < len(words) and words[k] not in _FUNC and words[k] not in _NOT_NOUN and words[k] not in _ADJ and len(words[k]) > 2 \
                        and not _VERBISH.search(words[k]) and not (words[k].endswith("s") and not words[k].endswith("ss")) and words[k] not in _STOPISH:
                    c = words[k]; k += 1
                best = f"{det} {c}"
    return best


_SLANG = set("lol lmao haha hahaha tho though tbh idk ikr omg btw anyway".split())
_FUNC = set("""a an the of in on at to for with from by about into over under after before between and or but so because if when while that which who whom whose is was are were be been am
do does did has have had will would can could should may might must i you he she it we they me him her us them my your his its our their this these those there here not no yes too very
just only also again then now than as up down out off away back goes never always often sometimes usually rarely ever already still even almost nearly without within through across during around against toward towards onto upon like per via unless until since while although though however whether
""".split())
_STOPISH = set("""what that this with have they them then than there about would could should really very just like your mine been were will from into some more
who whom whose when where which while have does done not but and for are was his her its our out off too than also only""".split())


def is_claim(msg, n):
    """A real statement (not a fragment/exclamation/question) long enough to talk about."""
    if msg.rstrip().endswith("?"):
        return False
    w = n.split()
    return len(w) >= 4 and bool(re.match(r"(?:i|i'm|im|i've|ive|my|we|our|the|it|it's|its|she|he|they|there|this|that|one|some|just|so|yeah|and|but|anyway|then|today|yesterday|last|now|when|after|before|because)\b", n))


ACK_POOL = [
    "I see. How are you feeling about all that?",
    "Thanks for telling me. Is there more to it?",
    "Okay, I'm listening. Say more if you like.",
    "That makes sense. What else is on your mind?",
]
TOPIC_POOL = [
    "Tell me more about {t}.",
    "How do you feel about {t}?",
    "What's on your mind about {t}?",
]
FRAG_POOL = [
    "Okay, I'm listening.", "Go on, I'm listening.", "I see. What else is on your mind?", "Hm, interesting. Tell me more when you're ready.",
]
PLAN = re.compile(r"\b(?:tomorrow|tonight|next (?:week|month|june|monday|year|friday)|on (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|i start|i'?ll|i'm going to|im going to|i will|going to|we'?ll|gonna|later)\b")
WRONGTOPIC = {"name", "way"}


def backchannel(msg, n, used, turn):
    """Context-aware but safe reply to a statement nothing else handled."""
    claim = is_claim(msg, n)
    plan = PLAN.search(n)
    if plan and len(n.split()) >= 3 and not topic_noun(msg):
        return _pick(["Sounds like you have a plan. How are you feeling about it?",
                      "Okay, I'm listening. What would you like to happen?"], used, "plan", turn)
    if claim:
        t = topic_noun(msg)
        if plan and not t:
            return _pick(["That sounds like something to look forward to. How are you feeling about it?",
                          "Sounds like you have plans. What are you hoping for?"], used, "plan", turn)
        if t and t.split()[-1] not in WRONGTOPIC:
            return _pick([p.format(t=t) for p in TOPIC_POOL], used, "topic", turn)
        return _pick(ACK_POOL, used, "ack", turn)
    if len(n.split()) >= 3:
        return _pick(ACK_POOL, used, "ack", turn)
    return _pick(FRAG_POOL, used, "frag", turn)


# ---------------------------------------------------------------- acknowledgements
ACK_WORDS = set("""ok okay k kk cool nice lol lmao rofl haha hahaha hehe heh yes yeah yep yup yea no nope nah sure hmm hm mm mhm wow thanks thank you thx ty great fine alright right
idk ikr fair either true same agreed indeed exactly oh ah aha ahh uh um well so then anyway deep interesting honest good gotcha got it ill i'll try that maybe perhaps whatever bruh dude man
omg oops oh ooh yay woo please just yet again lol. hahah ahaha lolol fr tho though now yes. noted nvm never mind sorry dunno k. haha. awesome neat sweet true that""".split())


NONANSWER = re.compile(r"^(?:idk|i don'?t know|i dunno|dunno|no idea|not sure|me neither|neither do i|same|i do not know|i have no idea|nothing|never mind|nvm|forget it|maybe|i guess)\b")


def is_nonanswer(n):
    return is_ack(n) or bool(NONANSWER.match(n))


def is_ack(n):
    w = n.split()
    return 0 < len(w) <= 4 and all(x in ACK_WORDS for x in w)


# ---------------------------------------------------------------- generic memory: pets / job / relationships
ANIMALS = "cat dog puppy kitten hamster rabbit bunny fish parrot bird horse turtle snake lizard guinea pig".split()
BREEDS = {"retriever": "dog", "labrador": "dog", "poodle": "dog", "bulldog": "dog", "terrier": "dog", "husky": "dog", "shepherd": "dog", "beagle": "dog", "spaniel": "dog",
          "collie": "dog", "dachshund": "dog", "chihuahua": "dog", "pug": "dog", "corgi": "dog", "siamese": "cat", "tabby": "cat", "persian": "cat", "kitty": "cat", "puppy": "dog",
          "kitten": "cat", "bunny": "rabbit"}
ANIMAL_RE = r"(?:retrievers?|labradors?|poodles?|bulldogs?|terriers?|huskys?|shepherds?|beagles?|spaniels?|collies?|dachshunds?|chihuahuas?|pugs?|corgis?|siamese|tabbys?|persians?|cats?|dogs?|puppy|puppies|kittens?|kitty|hamsters?|rabbits?|bunny|bunnies|fish|parrots?|birds?|horses?|turtles?|snakes?|lizards?|guinea pigs?|pets?)"
RELS = r"(?:sister|brother|mom|mum|mother|dad|father|wife|husband|girlfriend|boyfriend|partner|fianc[eé]e?|friend|best friend|son|daughter|kid|boss|grandma|grandpa|grandmother|grandfather|aunt|uncle|cousin|roommate|teacher|dog|cat|pet)"
NUMWORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
OCC = ("teacher nurse doctor engineer developer programmer designer student lawyer chef cook driver artist writer musician accountant manager electrician plumber "
       "mechanic farmer pilot dentist scientist researcher professor librarian waiter waitress barista cashier banker analyst consultant carpenter painter photographer "
       "firefighter police officer soldier journalist actor actress baker pharmacist therapist builder tailor vet veterinarian salesperson freelancer entrepreneur").split()
CAPNAME = r"[A-Z][a-zA-Z'À-ɏ-]+"


def singular(w):
    w = w.lower()
    for a, b in (("puppies", "puppy"), ("bunnies", "bunny"), ("guinea pigs", "guinea pig"), ("huskies", "husky")):
        if w == a:
            return b
    return w[:-1] if w.endswith("s") and not w.endswith("ss") and w != "fish" else w


def names_list(s):
    return [x for x in re.findall(CAPNAME, s) if x.lower() not in ("and", "i", "the", "my")]


def join_names(ns):
    return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]
