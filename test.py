from lingua import Language, LanguageDetectorBuilder

languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

fulltext = """War bey Fürst Clary in T〈e〉plitz die Rede von Büchern und
lesen; da fragte Baron Eckartstein die Frau von Lubinska ob sie
Goethen's Wanderjahre schon gelesen habe. Nein sagte sie, ich
habe sie aber, und werde sie alsbald anfangen.
pourquoi lire de [pareil] choses
pareilles? Sprach Fürst Solms – unser General – drein;
l'on voit tout de suite que cet homme n'a
jamais fréquenté la bonne société; et quel monde il a
vu.

Ekartstein erzehlte es mir gleich nach her; und wollte aus der
Haut fahren; und ich schreibe es auf weil so etwas nicht verlohren
gehen muß; [und] Ekarst: gab mir sein Ehrenworth daß es
buchstäblich so übereinanderging, weil ich zweiflen wollte. Auch
hatte er schon als Eckartst: den Herzog von Waimar zu kennen
wünschte und hinzusetzte schon deswegen, weil er so große
Tallente um sich gesammelt habe, und Goethen zum Freund hat,
[sagte] gesagt, der Herzog habe
mauvais ton,
et que l'on voyait tout de suite qui il
frequente. etc:.""".replace("\n", " ").replace("\r", " ")


def white_space_tokenise(text: str):
    for part in text.split():
        yield part


WHITESPACE_TOKENS = " \n\r\t\f"
PUNCTUATION_TOKENS = ".,:;-?![](){}"


def tokenise(text: str):
    buffer = []
    for char in text:
        if char in WHITESPACE_TOKENS:
            if len(buffer) > 0:
                yield "".join(buffer)
                buffer = []
            yield char
        elif char in PUNCTUATION_TOKENS:
            if len(buffer) > 0:
                yield "".join(buffer)
                buffer = []
            yield char
        else:
            buffer.append(char)
    if len(buffer) > 0:
        yield "".join(buffer)


current_lang = None
buffer = []
for token in tokenise(fulltext):
    if token in WHITESPACE_TOKENS:
        buffer.append(token)
    elif token in PUNCTUATION_TOKENS:
        buffer.append(token)
    else:
        lang = detector.detect_language_of(token)
        if lang != current_lang:
            if len(buffer) > 0:
                print(current_lang, "".join(buffer))
                buffer = []
            current_lang = lang
        buffer.append(token)
print()

result = []
for token in tokenise(fulltext):
    if token in WHITESPACE_TOKENS:
        result.append((token, None))
    elif token in PUNCTUATION_TOKENS:
        result.append((token, None))
    else:
        result.append(
            (
                token,
                [conf for conf in detector.compute_language_confidence_values(token)],
            )
        )

for token, languages in result:
    if languages is not None:
        print(
            token,
            [(conf.language.iso_code_639_1.name, conf.value) for conf in languages],
        )
    else:
        print(token)

for detection in detector.detect_multiple_languages_of(fulltext):
    print(detection, fulltext[detection.start_index : detection.end_index])
