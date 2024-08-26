import win32com.client as client
from structure import Structure
import json
from pathlib import Path

PATH = r"/VKR/mycomp/Taxfacto/JSON скриптер/Подборка решений по TaxFacto"

def add_value(obj, key, text, type_=''):
    value = obj[key]
    if isinstance(value, str):
        obj[key] += text
    elif isinstance(value, dict):
        keys = list(value.keys())
        value[keys[0]] = text
        value[keys[1]] = type_
    else:
        if isinstance(value[-1], str):
            val_copy = ""
        else:
            val_copy = value[-1].copy()
        add_value(value, -1, text, type_)
        value.append(val_copy)


def comments_to_json(comments, structure: Structure):
    s = structure.copy()

    for c in comments:
        if c.Ancestor is None:
            key = c.Range.Text.lower()
            if key in structure.comments_to_keys:
                add_value(s, structure.comment_to_key(key), c.Scope.Text)
            else:
                key_parts = key.split()
                if key_parts[0] in structure.comments_to_keys:
                    if "налогоплательщик" in key:
                        add_value(s, structure.comment_to_key(key_parts[0]), c.Scope.Text, "Налогоплательщик")
                    elif "орган" in key:
                        add_value(s, structure.comment_to_key(key_parts[0]), c.Scope.Text, "Налоговый орган")
                    else:
                        add_value(s, structure.comment_to_key(key_parts[0]), c.Scope.Text)
    return s


def process_file(fname):
    doc = word.Documents.Open(fname+'.docx')
    doc.Activate()
    active_doc = word.ActiveDocument

    structure = Structure()
    result = comments_to_json(active_doc.Comments, structure)
    doc.Close()

    Structure.clear(result)
    # Structure.add_articles(result)

    with open(fname + '.json', 'w', encoding='utf-8') as f:
        json.dump(result, f)


word = client.gencache.EnsureDispatch('Word.Application')
word.Visible = False


p = Path(PATH)
for full_path in p.rglob("*"):
    full_path_s = str(full_path)
    if 'docx' in full_path_s:
        print("Processing " + full_path_s + "...")
        process_file(full_path_s[:-5])
        print("Done!")
