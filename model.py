UWU_DICT = {
    # greetings
    "hello": "hewwo",
    "hi": "haiii",
    "hey": "heyyy",
    "good morning": "gud mowning",
    "good night": "gud nyight",
    "welcome": "wewcome",
    "bye": "bai",
    "goodbye": "buh-bai",
    "see you": "see uu",
    "later": "watew",

    # pronouns
    "you": "uu",
    "your": "ur",
    "yours": "urs",
    "you're": "ur",
    "you are": "uu awe",
    "me": "mwe",
    "my": "muh",
    "mine": "miney",
    "we": "whe",
    "us": "usuu",
    "our": "ouw",

    # politeness
    "please": "pwease",
    "thank you": "fank uu",
    "thanks": "fanks",
    "sorry": "sowwy",
    "excuse me": "scuse mwe",
    "welcome": "uwu-welcome",

    # emotions
    "love": "wuv",
    "like": "wike",
    "hate": "hatey-watey",
    "happy": "happi",
    "sad": "saddy",
    "angry": "angy",
    "mad": "maddie",
    "scared": "scawwed",
    "afraid": "afwaid",
    "excited": "egg-cited",
    "bored": "bowed",
    "tired": "tywed",
    "sleepy": "sweepy",
    "lonely": "wonewy",

    # descriptors
    "small": "smol",
    "little": "widdle",
    "big": "biggo",
    "cute": "kawaii",
    "pretty": "pwetty",
    "ugly": "ugwy",
    "funny": "funni",
    "serious": "sewious",
    "weird": "wheewd",
    "nice": "nyice",
    "mean": "meanie",
    "good": "gud",
    "bad": "baddie",
    "cool": "kewl",
    "awesome": "awesum",

    # verbs
    "stop": "stawp",
    "go": "gow",
    "come": "cum",
    "wait": "waity",
    "run": "wun",
    "walk": "wawk",
    "eat": "eet",
    "drink": "dwink",
    "sleep": "sweep",
    "think": "fink",
    "know": "knu",
    "want": "wunt",
    "need": "needy",
    "have": "haz",
    "make": "maek",
    "look": "wook",
    "see": "see",
    "say": "sayy",
    "tell": "teww",
    "help": "hewp",
    "try": "twy",
    "play": "pway",
    "work": "wowk",

    # questions / misc
    "what": "wut",
    "why": "wy",
    "who": "hoo",
    "when": "wen",
    "where": "whewe",
    "how": "howo",
    "this": "dis",
    "that": "dat",
    "these": "dese",
    "those": "dose",
    "here": "hewe",
    "there": "dewe",

    # responses
    "yes": "yus",
    "yeah": "ye",
    "yep": "yupii",
    "no": "nu",
    "nope": "nuh-uh",
    "maybe": "maybey",
    "ok": "oki",
    "okay": "okii",
    "alright": "awight",

    # swears / softeners
    "god": "gosh",
    "damn": "darn",
    "hell": "heck",
    "fuck": "fwick",
    "shit": "shoot",
    "crap": "cwap",

    # internet / slang
    "lol": "hehe",
    "lmao": "hehehehe",
    "rofl": "woll",
    "omg": "omggies",
    "wtf": "wut da fwip",
    "idk": "i dunno",
    "btw": "btwuu",
    "pls": "pwease",
    "thx": "fanks",

    # time / quantity
    "now": "nyow",
    "soon": "soony",
    "later": "watew",
    "always": "awways",
    "never": "nevew",
    "everything": "evewything",
    "nothing": "nuffin",

    # tech / fun extras
    "computer": "compuwter",
    "server": "sewvew",
    "model": "moduw",
    "api": "appy-i",
    "code": "cowde",
    "python": "pythyon",
    "error": "ewwow",
    "bug": "bwig",
    "timbur": "rico",
    "loves" :"wuvs",
    "love": "wuv"
}
text = "Hello Sir, how are you doing?"

def encode(text):
    ans = []
    for word in text.split():
        if word in UWU_DICT:
            ans.append(UWU_DICT[word])
        else:
            ans.append(word)
    
    return " ".join(ans)
