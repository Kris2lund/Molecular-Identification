import re
import time
import random
import json
import math
import csv
import threading
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from google import genai
from openai import OpenAI
from anthropic import Anthropic

from ase import Atoms
from ase.io import write

# ============================================================
# SETTINGS
# ============================================================
# Definationer af modeller, Kald API, og run-settings
# ============================================================

# Definering af LLM modellerne
MODEL_NAME_GEMINI = "gemini-2.5-flash"                  
MODEL_NAME_OPENAI = "gpt-5.4"
MODEL_NAME_CLAUDE = "claude-sonnet-4-5-20250929"

# Kald Gemini via Vertex AI (Google Cloud)
LOCATION = "europe-west4"
PROJECT_ID = "project-82aa0380-4f09-4372-bb3"

# Kald af PubChem basen via API
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Antal molekyler at udvælge og behandle i pipelinen
TARGET_N = 100

# Hvor mange PubChem-navne der må testes parallelt under udvælgelse
MAX_WORKERS_PUBCHEM_SELECT = 12

# Hvor mange molekyler der må behandles parallelt for hver LLM og PubChem 3D
MAX_WORKERS_MOLECULES = 4

# Hvor mange kald pr. molekyle der må køre samtidig
# (Gemini + OpenAI + Claude + PubChem = 4)
MAX_WORKERS_PER_MOLECULE = 4

# Venter max 15 sekunder på PubChem properties og 30 sekunder på 3D struktur LLM
REQUEST_TIMEOUT_PROPS = 15
REQUEST_TIMEOUT_3D = 30

# Hvis du vil spare tid, kan du slå xyz fra
WRITE_XYZ = True

# Kør hele pipelinen automatisk efter generering videre til sammenligning
RUN_COMPARISON_AFTER_GENERATION = True
SYNC_TO_PROJECT_ROOT = False

# ============================================================
# OUTPUT PATHS
# ============================================================
# Opretter paths til data og strukturer, og definerer hvor output CSV og JSON filer skal gemmes
# Hver kørsel gemmes i sin egen run-mappe, så tidligere datasæt ikke overskrives
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
DATA_DIR = None
STRUCT_ROOT = None
STRUCT_GEMINI = None
STRUCT_OPENAI = None
STRUCT_CLAUDE = None
STRUCT_PUBCHEM = None
GEMINI_CSV = None
OPENAI_CSV = None
CLAUDE_CSV = None
PUBCHEM_CSV = None
FINAL_LIST_JSON = None
CURRENT_RUN_DIR = None
CURRENT_RUN_ID = None

# Finder næste run-nummer og opretter en ny run-mappe (run_001, run_002, ...)
def get_next_run_dir():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing_ids = []
    for p in RUNS_DIR.iterdir():
        if p.is_dir():
            m = re.fullmatch(r"run_(\d{3})", p.name)
            if m:
                existing_ids.append(int(m.group(1)))
    next_id = max(existing_ids, default=0) + 1
    run_dir = RUNS_DIR / f"run_{next_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, next_id

# Opretter nødvendige paths for den aktuelle kørsel og definerer output paths
def setup_run_paths():
    global DATA_DIR, STRUCT_ROOT, STRUCT_GEMINI, STRUCT_OPENAI, STRUCT_CLAUDE, STRUCT_PUBCHEM
    global GEMINI_CSV, OPENAI_CSV, CLAUDE_CSV, PUBCHEM_CSV, FINAL_LIST_JSON
    global CURRENT_RUN_DIR, CURRENT_RUN_ID

    CURRENT_RUN_DIR, CURRENT_RUN_ID = get_next_run_dir()
    DATA_DIR = CURRENT_RUN_DIR / "data"
    STRUCT_ROOT = CURRENT_RUN_DIR / "structures"

    # Opret underpaths for hver source (Gemini, OpenAI, Claude, PubChem)
    STRUCT_GEMINI = STRUCT_ROOT / "gemini"
    STRUCT_OPENAI = STRUCT_ROOT / "openai"
    STRUCT_CLAUDE = STRUCT_ROOT / "claude"
    STRUCT_PUBCHEM = STRUCT_ROOT / "pubchem"

    # Opretter mappen til data og strukturer for den aktuelle kørsel
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STRUCT_GEMINI.mkdir(parents=True, exist_ok=True)
    STRUCT_OPENAI.mkdir(parents=True, exist_ok=True)
    STRUCT_CLAUDE.mkdir(parents=True, exist_ok=True)
    STRUCT_PUBCHEM.mkdir(parents=True, exist_ok=True)

    # Opret paths for output CSV filer og JSON fil til den endelige liste af udvalgte molekyler
    GEMINI_CSV = DATA_DIR / "molecules_gemini.csv"
    OPENAI_CSV = DATA_DIR / "molecules_openai.csv"
    CLAUDE_CSV = DATA_DIR / "molecules_claude.csv"
    PUBCHEM_CSV = DATA_DIR / "molecules_pubchem.csv"
    FINAL_LIST_JSON = DATA_DIR / "final_selection_pubchem_100.json"

# Synkroniserer den seneste kørsel til projektets hovedmapper, så næste pipeline-trin kan læse data som før
# Tidligere indhold i hovedmapperne slettes kun ved synkronisering af den aktuelle kørsel
# Historikken i runs/ bevares altid
def sync_latest_run_to_project_root():
    if not SYNC_TO_PROJECT_ROOT:
        print("Skipping sync to project root.")
        return

# Initialiserer paths for denne kørsel med det samme, så resten af scriptet kan bruge dem som normalt
setup_run_paths()

# Printer en besked for at indikere, at processen er startet
print(f"Generating LLM + PubChem datasets -> CSV ... (run_{CURRENT_RUN_ID:03d})")

# ============================================================
# GLOBALS / THREAD-LOCAL CLIENTS
# ============================================================
# Opret sperat local thread storage, definer regex mønster for JSON, colum headers for CSV, og mapping fra atomnummer til symbol
# ============================================================

# Giver hver tråd sit eget lokale lager til clients og sessioner
thread_local = threading.local()

# Definerer et regex mønster til at finde JSON i output fra LLM'erne
# Re.compile => Preperes regex mønster
# r"\{.*\}" => Matcher alt i {...}
# re.DOTALL => Matcher også med newline
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Column headers for CSV output
FIELDNAMES = [
    "cid", "name_key",
    "chemical_name", "chemical_formula", "atom_count_total",
    "atom_count_by_element"
]

# Mapping fra atomnummer til symbol, Pubchem (atom nummer) => output (symbol)
ATOMIC_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    35: "Br", 53: "I"
}

# ============================================================
# CLIENT FACTORIES
# ============================================================
# Opretter check for celienter/sessioner i tråde, og opretter dem hvis de ikke findes samt retry strategier for requests session til PubChem
# ============================================================

# Checker om session til tread findes, hvis ikke, oprettes en ny session 
# Primært for de mange PubChem kald og retry strategi for at håndtere netværksfejl og rate limits
def get_requests_session():
    if not hasattr(thread_local, "session"):
        session = requests.Session() # Opretter genbrugelig HTTP session

        # Definer retry strategi for requests session, så den automatisk prøver igen (defineret tries pr. fejl)
        retry = Retry(
            total=3,
            read=3,
            connect=3,
            backoff_factor=0.6, # Ventetid mellem retries adderet
            status_forcelist=[429, 500, 502, 503, 504], # HTTP status koder der trigger retry
            allowed_methods=["GET"]
        )

        # Adapter med connection pooling og retry strategi
        # Pool_conections => Antal connection pools at oprette
        # Pool_maxsize => Maks antal connections i poolen
        # Max_retries => Retry strategi defineret i "retry"-objektet
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retry)
        session.mount("http://", adapter)  # session.mount => Brug denne adapter for URL der starter med http:// og https://
        session.mount("https://", adapter)
        thread_local.session = session # Gem session i thread-local storage for genbrug i samme thread
    return thread_local.session

# Chekker om Gemini client findes i thread, ellers oprettes den
def get_gemini_client():
    if not hasattr(thread_local, "gemini_client"):
        thread_local.gemini_client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION
        )
    return thread_local.gemini_client

# Checker om OpenAI client findes i thread, ellers oprettes den
def get_openai_client():
    if not hasattr(thread_local, "openai_client"):
        thread_local.openai_client = OpenAI()
    return thread_local.openai_client

# Checker om Claude client findes i thread, ellers oprettes den
def get_claude_client():
    if not hasattr(thread_local, "claude_client"):
        thread_local.claude_client = Anthropic()
    return thread_local.claude_client

# ============================================================
# HELPERS
# ============================================================
# Forbereder filnavne, normaliserer molekyle navne, finder JSON block i tekst, omdanner JSON data til ASE Atoms objekter, tæller atomer,
# og forbereder data til CSV output
# ============================================================

# Forebereder filnavne navn (igen special tegn)
def safe_filename(name: str) -> str:
    name = str(name).strip().lower() # Tekst, -mellemrum, lowercase
    name = re.sub(r"[^\w\-]+", "_", name) # Erstat specialtegn med underscore
    return name[:120] if len(name) > 120 else name # Grænse for længde på navn

# Opret key navn (molekyle navne) nomilseret (Tekst, -mellemrum, lowercase)
def normalize_name_key(name: str) -> str:
    return str(name).strip().lower()

# Finde JSON block i tekst med "regex", returnerer den eller None hvis ikke fundet
def extract_json_from_text(text: str):
    if not text:
        return None
    m = JSON_RE.search(text)
    return m.group(0) if m else None

# Omdanner JSON data til ASE Atoms Objekter, og tilføjer varccum omkring molekylerne for at sikre korrekt håndtering i ASE og senere sammenligning
def json_to_ase_atoms(mol_json, vacuum=3.0) -> Atoms: # mol_jason (input), vacuum omkring molekylet, returnerer ASE Atoms objekt
    symbols = [a["element"] for a in mol_json["atoms"]] # Henter "element" i "atoms" listen i JSON dataen
    positions = [[a["x"], a["y"], a["z"]] for a in mol_json["atoms"]] # Henter koordinaterne i "atoms" listen i JSON dataen

    #Defininerer ASE atom
    atoms = Atoms(symbols=symbols, positions=positions) 

    # Find molekylets udstrækning
    pos = atoms.get_positions()
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    span = maxs - mins

    # Lav celle som molekylets størrelse + vacuum på begge sider
    cell = span + 2 * vacuum
    atoms.set_cell(cell)

    # Center molekylet i den nye celle
    atoms.center()

    # Ingen periodiske randbetingelser for isoleret molekyle
    atoms.set_pbc([False, False, False])

    return atoms

# Tæller atomer i hver molekyle
def count_elements(atoms_list):
    counts = {}
    for a in atoms_list:
        el = a["element"]
        counts[el] = counts.get(el, 0) + 1
    return counts

# Checker om LLM atom count matcher forventet atom count fra PubChem formel
def validate_atom_count(mol, expected_atom_count):
    llm_atom_count = len(mol.get("atoms", []))
    expected_atom_count = int(expected_atom_count or 0)

    if expected_atom_count <= 0:
        raise ValueError("Missing expected PubChem atom count")

    if llm_atom_count != expected_atom_count:
        raise ValueError(
            f"Atom count mismatch. Expected {expected_atom_count}, got {llm_atom_count}"
        )

# Bestemmer total mængde atomer ud fra kemisk formel (f.eks. C6H12O6 => 24 atomer)
def atom_count_from_formula(formula: str) -> int:
    if not formula:
        return 0
    total = 0
    for _, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula): # Definerer regex møster Stort + Lille + Tal
        total += int(num) if num else 1
    return total

# Output retunerer tomme rows ved fejl fra Pubchem eller LLM til CSV
def empty_row(cid, name_key, forced_name):
    return {
        "cid": cid,
        "name_key": name_key,
        "chemical_name": forced_name,
        "chemical_formula": "",
        "atom_count_total": "",
        "atom_count_by_element": "",
    }


def mol_to_csv_row(mol_dict, cid, name_key, forced_name):
    atoms_list = mol_dict.get("atoms", []) # Henter "atoms" listen fra mol_dict, eller tom liste hvis ikke findes
    atom_count_by_el = count_elements(atoms_list) # Tæller atomer i "atoms" listen og grupperer dem efter element (f.eks. {"C": 6, "H": 12, "O": 6} for C6H12O6)
    atom_count_total = len(atoms_list) # Tæller total mængde atomer i "atoms" listen

    # Returnerer relevante data som en dictionary, der kan skrives til CSV
    # json.dumps => Gem kompleks data (liste eller dict) som string i CSV
    # ensure_ascii=False => Bevarer unicode karakterer i output (fx. "æøå" i kemiske navne)
    return {
        "cid": cid,
        "name_key": name_key,
        "chemical_name": forced_name,
        "chemical_formula": mol_dict.get("chemical_formula", ""),
        "atom_count_total": atom_count_total,
        "atom_count_by_element": json.dumps(atom_count_by_el, ensure_ascii=False),
    }

# ============================================================
# PUBCHEM name -> props
# ============================================================
# Slår molekylenavne op i PubChem, bygger en kandidatliste og udvælger unikke CIDs via parallelle opslag.
# ============================================================

# Henter egenskaber for et molekyle baseret på dets navn i PubChem
def fetch_props_by_name(name: str):
    session = get_requests_session() # Kalder PubChem-session kode fra thread-local
    props = "MolecularFormula,MolecularWeight,IUPACName,IsomericSMILES,HeavyAtomCount" # Definerer egenskaber til finding i Pubchem
    url = f"{PUBCHEM}/compound/name/{requests.utils.quote(name)}/property/{props}/JSON" # Kalder adreasse til specifict opslag i Pubchem


    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT_PROPS) # Kalder get på dataen fra PubChem med timeout
        if r.status_code != 200: # HTTP2/200 => Get var succesfuld, ellers returneres None
            return None

        data = r.json() # Jason data fra PubChem omdannes til Python dict
        rows = data.get("PropertyTable", {}).get("Properties", []) # Hent "Properties" listen fra "PropertyTable" i dataen, eller tom liste hvis ikke findes
        if not rows:
            return None

        row = rows[0] # Kun den første proberties retuneret bruges
        formula = row.get("MolecularFormula", "") or "" # Hent "MolecularFormula" fra row, eller tom string hvis ikke findes

        # Opsætning af stanndartiseret output format fra Pubchem, som senere kan bruges i prompt og sammenligning
        return {
            "query_name": name, # Molekyle navn brugt i søgning i PubChem
            "cid": row.get("CID", ""), # Compond ID fra PubChem
            "iupac_name": row.get("IUPACName", "") or "", # Kemisk navn i IUPAC format fra PubChem
            "formula": formula, # Kemisk formel fra PubChem
            "molecular_weight": row.get("MolecularWeight", ""), # Molekylær vægt fra PubChem
            "heavy_atom_count": row.get("HeavyAtomCount", ""), # Antal tunge atomer (ikke-hydrogen) fra PubChem
            "atom_count_est": atom_count_from_formula(formula), # Estimeret total mængde atomer ud fra den kemiske formel
            "isomeric_smiles": row.get("IsomericSMILES", "") or "", # Isomeric SMILES notation fra PubChem, som beskriver molekylets struktur i tekstformat
        }
    except requests.RequestException: # Ved netværksfejl eller timeout, returneres None
        return None

# Bygger en liste af molekyle navne til at søge i PubChem, og blander listen for at få en tilfældig rækkefølge
def build_candidate_names():
    base = [
        "water","carbon dioxide","methane","ammonia","hydrogen peroxide",
        "nitrogen","oxygen","ozone","carbon monoxide",
        "nitrous oxide","sulfur dioxide","hydrogen sulfide","hydrogen chloride",
        "hydrogen bromide","hydrogen iodide","hydrogen fluoride",
        "formaldehyde","acetaldehyde","acetone","methanol","ethanol","propanol",
        "isopropanol","acetic acid","formic acid","propionic acid",
        "dimethyl ether","diethyl ether","ethyl acetate",
        "benzene","toluene","aniline","phenol","pyridine",
        "urea","glycine","alanine","valine","leucine",
        "glucose","fructose","sucrose",
        "chloroform","dichloromethane","carbon tetrachloride",
        "acetonitrile","dimethyl sulfoxide","tetrahydrofuran",
        "cyclohexane","cyclopentane","ethylene","propene","1-butene",
        "acetylene","1,3-butadiene",
        "ethylamine","diethylamine","triethylamine",
        "methyl acetate","ethyl formate","ethyl propionate",
        "acetamide","benzamide",
        "piperidine","morpholine",
        "naphthalene","styrene","ethylbenzene",
        "thiophene","furan",
        "hydrazine","hydroxylamine",
    ]
    alkanes = ["ethane","propane","butane","pentane","hexane","heptane","octane","nonane","decane","undecane","dodecane"]
    alcohols = ["1-propanol","2-propanol","1-butanol","2-butanol","tert-butanol","1-pentanol","1-hexanol","1-heptanol","1-octanol"]
    acids = ["butyric acid","valeric acid","caproic acid","benzoic acid","salicylic acid"]
    halides = ["chloromethane","bromomethane","iodomethane","chloroethane","bromoethane","1-chloropropane","2-chloropropane","1-chlorobutane","1-bromobutane","1-iodobutane"]

    # Retunrner lister sammen til en lang blandet liste og fjerner dublikater
    candidates = list(dict.fromkeys(base + alkanes + alcohols + acids + halides))
    random.shuffle(candidates)
    return candidates

# Udvælger unikke CIDs ved at søge i PubChem med navne fra build_candidate_names, og bruger ThreadPoolExecutor til at gøre det parallelt 
def select_unique_cids(target_n=100, max_workers=12):
    candidates = build_candidate_names() # Molekyler vælges fra liste ved tidligere funktion 
    selected = [] 
    seen_cids = set() 

    # ThreadPoolExecutor bruges til at køre flere fetch_props_by_name funktioner parallelt
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_props_by_name, name): name for name in candidates} # Definer futures for hver kandidat i listen, hvor fetch_props_by_name kaldes parallelt

        for fut in as_completed(futures): # Run gennem futures efterhånden som de bliver færdige
            if len(selected) >= target_n: # Break hvis traget_n (100) er nået
                break

            res = fut.result() # Henter funture resultatet
            if not res:
                continue

            cid = str(res.get("cid", "")).strip() # Henter CID fra resultatet som string uden mellemrum
            if not cid or cid in seen_cids: # Chek om CID er set eller ikke-eksisterende
                continue

            seen_cids.add(cid)
            selected.append(res) # Indskriv molekylet i listen
            print(f"[{len(selected):3d}/{target_n}] OK: {res['query_name']} -> CID {cid} ({res['formula']})") # Udskriver status for udvælgelsen

    return selected[:target_n] # Retunerer de første target_n molekyler fra den udvalgte liste

# ============================================================
# PUBCHEM 3D by CID
# ============================================================
# Denne sektion henter 3D-strukturdata fra PubChem og omdanner dem til et standardiseret molekyle-format
# med kemisk formel, atomer og koordinater.
# ============================================================

# Bruger CID til at hente molekyle proberties, og retunerer det som dict
def fetch_pubchem_3d_by_cid(cid: str):
    session = get_requests_session() # Hetner request session for Pubchem
    url = f"{PUBCHEM}/compound/cid/{cid}/record/JSON?record_type=3d" # Bygger URL, til specifict CID, og finder 3D data i JSON-filen

    # Henter 3D data med timeout, og chekker for svarstatus
    r = session.get(url, timeout=REQUEST_TIMEOUT_3D)
    if r.status_code != 200:
        raise RuntimeError(f"PubChem 3D failed, status={r.status_code}")

    data = r.json() # Laver pubchem JSON til python data 
    record = data["PC_Compounds"][0] # Henter den første compound record fra dataen

    formula = "" # Empty string
    for prop in record.get("props", []): # Loop gennem "props" listen i record 
        urn = prop.get("urn", {}) # Henter "urn" dict fra prop, eller tom dict hvis ikke findes
        if urn.get("label") == "Molecular Formula": # Chekker om "label" i "urn" er "Molecular Formula"
            value = prop.get("value", {}) # Henter "value" dict fra prop, eller tom dict hvis ikke findes
            formula = value.get("sval") or value.get("fval") or "" # Gemmer formel som stringvalue, floatvalue eller tom string
            break

    atoms = record["atoms"] # Henter "atoms" dict fra record
    coords = record["coords"][0]["conformers"][0] # Henter koordinaterne for det første konformer i "coords" listen i record
    x, y, z = coords["x"], coords["y"], coords["z"] # Henter x, y, z koordinaterne fra coords

    # Struktur for og output fra PubChem
    out = {
        "chemical_name": "",
        "chemical_formula": formula,
        "atoms": []
    }

    # Loop gennem atomerne i "atoms" dict, og tilføjer dem til "atoms" listen i output strukturen med element og koordinater
    for i, anum in enumerate(atoms["element"]):
        out["atoms"].append({
            "element": ATOMIC_SYMBOLS.get(anum, str(anum)),
            "x": float(x[i]),
            "y": float(y[i]),
            "z": float(z[i])
        })

    return out

# ============================================================
# PROMPT
# ============================================================
# Bygger en klar og struktureret prompt til LLM'erne, der specificerer kravene til output og inkluderer molekyle informationen
# ============================================================

# Bygger prompten til LLM'erne baseret på molekyle informationen, og specificerer klart kravene til output
def build_prompt(qname, cid, formula, smiles, iupac):
    return f"""
Generate ONLY valid JSON describing the molecular structure.

Target molecule:
- common_name: {qname}
- PubChem CID: {cid}
- molecular_formula: {formula}
- isomeric_smiles: {smiles}
- iupac_name (hint): {iupac}

Required fields (exact keys):
- chemical_name
- chemical_formula
- atoms: list of objects with fields: element, x, y, z

Rules:
- Output ONLY JSON. No text.
- chemical_formula MUST match the given molecular_formula exactly: {formula}
- The connectivity should be consistent with the isomeric_smiles when provided.
""".strip()

# ============================================================
# LLM CALLS
# ============================================================
# Definerer funktioner til at kalde hver LLM (Gemini, OpenAI, Claude) med den byggede prompt, 
# og håndterer output ved at udtrække JSON og omdanne det til det ønskede format
# ============================================================

# Sender prompt til Gemini og returnerer det rå tekst output, som senere skal parses for JSON
def generate_json_gemini(prompt: str) -> str: # String input prompt, string output
    client = get_gemini_client() # Client kaldes
    resp = client.models.generate_content( # Beder LLM om at generere output baseret på prompten
        model=MODEL_NAME_GEMINI, # Specificerer model
        contents=prompt # Kontekst er promten bygget tidligere
    )
    return resp.text or ""

# Sender prompt til OpenAI og returnerer det rå tekst output, som senere skal parses for JSON
def generate_json_openai(prompt: str) -> str:
    client = get_openai_client()
    resp = client.responses.create(
        model=MODEL_NAME_OPENAI,
        input=prompt,
    )
    return resp.output_text or ""

# Sender prompt til Claude og returnerer det rå tekst output i blocks, som senere skal parses for JSON
def generate_json_claude(prompt: str) -> str:
    client = get_claude_client()
    msg = client.messages.create(
        model=MODEL_NAME_CLAUDE,
        max_tokens=2000, 
        messages=[{"role": "user", "content": prompt}],
    )
    text = ""
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            text += block.text
    return text

# ============================================================
# SINGLE-SERVICE PROCESSORS
# ============================================================
# Definerer funktioner til at håndtere hver service (Gemini, OpenAI, Claude, PubChem) individuelt, 
# hvor hver funktion tager sig af at kalde den relevante service, håndtere output, skrive XYZ filer hvis relevant, og forberede data til CSV output
# ============================================================

# Funktion til at skrive XYZ filer for molekyler, hvis WRITE_XYZ er sandt, og der er atomer i molekylet
def maybe_write_xyz(struct_dir: Path, fname: str, mol: dict):
    if not WRITE_XYZ: # Checker om LLM har skrevet XYZ output
        return
    atoms = mol.get("atoms", [])
    if not atoms: # Checker om der er atomer i molekylet fra LLM output
        return
    ase_atoms = json_to_ase_atoms({"atoms": atoms}) # Omdanner JSON data til ASE Atoms objekt
    write((struct_dir / fname).as_posix(), ase_atoms) # Skriver ASE Atoms objekt til XYZ fil i den relevante struktur mappe

# Skriver en failure-markør fil, så næste pipeline-trin stadig kan registrere at molekylet blev forsøgt men ikke fundet
def write_failure_marker(struct_dir: Path, fname: str, source: str, message: str):
    if not WRITE_XYZ:
        return
    marker_path = struct_dir / fname
    marker_path.write_text(
        f"FAILED\nsource={source}\nmessage={message}\n",
        encoding="utf-8"
    )

# Hele gemnini workflow for et molekyle
def process_gemini(item, prompt, forced_name, name_key, fname):
    cid = str(item["cid"]) #Henter Pubchem molekyle
    try:
        out_text = generate_json_gemini(prompt) # Kalder Gemini med prompten og henter det rå tekst output
        json_text = extract_json_from_text(out_text) # Udtrækker JSON block fra det rå tekst output, som forventes at indeholde molekyle informationen
        if not json_text:
            raise ValueError("No JSON object found in Gemini output")

        mol = json.loads(json_text) # Omdanner JSON text til Python dict
        mol["chemical_name"] = forced_name # Overskriver "chemical_name" i mol dict med det "forced_name" der er baseret på query name
        mol["chemical_formula"] = item.get("formula", "") or mol.get("chemical_formula", "") # Overskriver "chemical_formula" i mol dict med den formel der er i PubChem dataen, eller beholder den i mol dict hvis ikke findes i PubChem

        validate_atom_count(mol, item.get("atom_count_est"))

        maybe_write_xyz(STRUCT_GEMINI, f"{fname}_gemini.xyz", mol) # Skriver XYZ fil for Gemini output hvis WRITE_XYZ er sandt og der er atomer i mol dict
        row = mol_to_csv_row(mol, cid, name_key, forced_name) # Forbereder CVS row data fra mol dict, og tilføjer CID, name_key og forced_name
        return "gemini", row, None
    except Exception as e:
        write_failure_marker(STRUCT_GEMINI, f"{fname}_gemini.xyz", "gemini", str(e))
        return "gemini", empty_row(cid, name_key, forced_name), str(e)

# Hele OpenAI workflow for et molekyle
def process_openai(item, prompt, forced_name, name_key, fname):
    cid = str(item["cid"])
    try:
        out_text = generate_json_openai(prompt)
        json_text = extract_json_from_text(out_text)
        if not json_text:
            raise ValueError("No JSON object found in OpenAI output")

        mol = json.loads(json_text)
        mol["chemical_name"] = forced_name
        mol["chemical_formula"] = item.get("formula", "") or mol.get("chemical_formula", "")

        validate_atom_count(mol, item.get("atom_count_est"))

        maybe_write_xyz(STRUCT_OPENAI, f"{fname}_openai.xyz", mol)
        row = mol_to_csv_row(mol, cid, name_key, forced_name)
        return "openai", row, None
    except Exception as e:
        write_failure_marker(STRUCT_OPENAI, f"{fname}_openai.xyz", "openai", str(e))
        return "openai", empty_row(cid, name_key, forced_name), str(e)

# Hele Claude workflow for et molekyle
def process_claude(item, prompt, forced_name, name_key, fname):
    cid = str(item["cid"])
    try:
        out_text = generate_json_claude(prompt)
        json_text = extract_json_from_text(out_text)
        if not json_text:
            raise ValueError("No JSON object found in Claude output")

        mol = json.loads(json_text)
        mol["chemical_name"] = forced_name
        mol["chemical_formula"] = item.get("formula", "") or mol.get("chemical_formula", "")

        validate_atom_count(mol, item.get("atom_count_est"))

        maybe_write_xyz(STRUCT_CLAUDE, f"{fname}_claude.xyz", mol)
        row = mol_to_csv_row(mol, cid, name_key, forced_name)
        return "claude", row, None
    except Exception as e:
        write_failure_marker(STRUCT_CLAUDE, f"{fname}_claude.xyz", "claude", str(e))
        return "claude", empty_row(cid, name_key, forced_name), str(e)

# Hele Pubchem workflow for et molekyle
def process_pubchem(item, forced_name, name_key, fname):
    cid = str(item["cid"])
    try:
        ref = fetch_pubchem_3d_by_cid(cid)
        ref["chemical_name"] = forced_name

        maybe_write_xyz(STRUCT_PUBCHEM, f"{fname}_pubchem.xyz", ref)
        row = mol_to_csv_row(ref, cid, name_key, forced_name)
        return "pubchem", row, None
    except Exception as e:
        write_failure_marker(STRUCT_PUBCHEM, f"{fname}_pubchem.xyz", "pubchem", str(e))
        return "pubchem", empty_row(cid, name_key, forced_name), str(e)

# ============================================================
# MOLECULE PROCESSOR
# ============================================================
# Vælger et molekyle, bygger prompten, og kalder hver service (Gemini, OpenAI, Claude, PubChem) 
# parallelt for at hente og behandle data, skrive XYZ filer hvis relevant, og forberede data til CSV output
# ============================================================

# Håndterer hele workflowet for et enkelt molekyle, hvor hver service kaldes parallelt, og resultaterne samles i en dict der kan bruges til CSV output
def process_molecule(item): 
    # Definese relevant information about the molecule from PubChem data
    qname = item["query_name"]
    cid = str(item["cid"])
    formula = item.get("formula", "")
    smiles = item.get("isomeric_smiles", "")
    iupac = item.get("iupac_name", "")

    forced_name = qname
    name_key = normalize_name_key(qname)
    fname = safe_filename(forced_name)
    prompt = build_prompt(qname, cid, formula, smiles, iupac)

    print(f"\n===== START: {forced_name} (CID {cid}) =====")

    results = {} # Dict til resultater fra servicer

    # Definerer opgaver (molekyle) for hver service som lambda funktioner, der kan kaldes parallelt
    tasks = {
        "gemini": lambda: process_gemini(item, prompt, forced_name, name_key, fname),
        "openai": lambda: process_openai(item, prompt, forced_name, name_key, fname),
        "claude": lambda: process_claude(item, prompt, forced_name, name_key, fname),
        "pubchem": lambda: process_pubchem(item, forced_name, name_key, fname),
    }

    # ThreadPoolExecutor bruges til at køre hver service opgave parallelt, 
    # og as_completed bruges til at håndtere resultaterne efterhånden som de bliver færdige
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PER_MOLECULE) as ex:
        futures = {ex.submit(func): name for name, func in tasks.items()}

        # Venter på at hver service opgave bliver færdig, og håndterer resultaterne ved at gemme dem i results dict, og udskrive status for hver service
        for fut in as_completed(futures):
            name = futures[fut]
            _, row, err = fut.result()
            results[name] = row

            if err:
                print(f"{name.capitalize()} FAILED for {forced_name}: {err}")
            else:
                print(f"{name.capitalize()} OK for {forced_name}")

    print(f"===== DONE: {forced_name} =====")
    return {
        "index": None,
        "name": forced_name,
        "gemini": results.get("gemini", empty_row(cid, name_key, forced_name)),
        "openai": results.get("openai", empty_row(cid, name_key, forced_name)),
        "claude": results.get("claude", empty_row(cid, name_key, forced_name)),
        "pubchem": results.get("pubchem", empty_row(cid, name_key, forced_name)),
    }

# ============================================================
# CSV WRITER
# ============================================================
# Denne funktion gemmer de behandlede molekyledata som en CSV-fil ved at skrive kolonneoverskrifterne og alle rækkerne til den angivne filsti.
# ============================================================

def write_rows_to_csv(path: Path, rows): #Output path for CSV fil, og rækker af data som liste af dicts
    with open(path, "w", newline="", encoding="utf-8") as f: # Åbner filen i skrive-tilstand, med UTF-8 encoding for at håndtere specialtegn
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES) # Laver CVS Writer, der forventer dicts som input, og bruger FIELDNAMES som kolonne headers
        writer.writeheader() # Skriver kolonne headers til CSV filen
        writer.writerows(rows) # Skriver alle rækkerne til CSV filen, hvor hver række er en dict der matcher FIELDNAMES

# ============================================================
# MAIN
# ============================================================
# Hovedfunktionen der styrer hele workflowet: udvælger molekyler, 
# kører process_molecule parallelt for hver molekyle, og skriver resultaterne til CSV filer.
# ============================================================

def main():
    t0 = time.perf_counter() # Starttid

    print("\nSelecting ~100 robust PubChem molecules (unique CID) ...")

    # Vælger det molekyle, der vil blive arbejdet med i resten af pipeline ved at søge i PubChem 
    # med navne fra build_candidate_names, og udvælge unikke CIDs via parallelle opslag
    selected = select_unique_cids(
        target_n=TARGET_N,
        max_workers=MAX_WORKERS_PUBCHEM_SELECT
    )

    print(f"\nSelected {len(selected)} molecules (unique CIDs). Saving selection -> {FINAL_LIST_JSON}")

    # Gemmer molekyler og deres CID'er i en JSON fil
    FINAL_LIST_JSON.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Definerer den valgte liste at molekyler
    final_list = selected

    # Opretter en liste af tuples med index og molekyle data for hvert molekyle i final_list, 
    # så vi kan holde styr på rækkefølgen når vi kører parallelt
    indexed_items = [(idx, item) for idx, item in enumerate(final_list)]

    all_results = [None] * len(indexed_items)

    print(f"\nProcessing molecules in parallel with MAX_WORKERS_MOLECULES={MAX_WORKERS_MOLECULES} ...")

    # Kører process_molecule funktionen parallelt for hvert molekyle i indexed_items, og gemmer resultaterne i all_results listen baseret på index
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_MOLECULES) as ex: # Definerer ThreadPoolExecutor for at køre flere process_molecule funktioner parallelt, med max antal = maxworkers_molecules
        
        # Begynder at arbejde på hvert molekyle, og gemmer en future i futre_map dict
        # Refrer hvert resutlat til dets index i indexed_items, så vi kan holde styr på rækkefølgen
        # når resultaterne kommer tilbage, selvom de bliver færdige i tilfældig rækkefølge
        future_map = {
            ex.submit(process_molecule, item): idx
            for idx, item in indexed_items
        }

        # Giver resultaterne når de er færdige, og gemmer dem i indexeret rækkefølge i all_results listen
        for fut in as_completed(future_map):
            idx = future_map[fut]
            result = fut.result()
            result["index"] = idx
            all_results[idx] = result

    # Forbereder rækker til CSV output ved at udtrække data for hver service (Gemini, OpenAI, Claude, PubChem)
    gemini_rows = [r["gemini"] for r in all_results]
    openai_rows = [r["openai"] for r in all_results]
    claude_rows = [r["claude"] for r in all_results]
    pubchem_rows = [r["pubchem"] for r in all_results]

    # Skriver rækkerne til CSV filer for hver service ved at kalde write_rows_to_csv funktionen
    write_rows_to_csv(GEMINI_CSV, gemini_rows)
    write_rows_to_csv(OPENAI_CSV, openai_rows)
    write_rows_to_csv(CLAUDE_CSV, claude_rows)
    write_rows_to_csv(PUBCHEM_CSV, pubchem_rows)

    dt = time.perf_counter() - t0 # Beregner total runtime 

    # Udskriver status og relevante informationer om output filer, runtime, og strukturer
    print(
        f"\nDONE ✅\n"
        f"Runtime: {dt:.1f} s\n"
        f"Gemini CSV: {GEMINI_CSV}\n"
        f"OpenAI CSV: {OPENAI_CSV}\n"
        f"Claude CSV: {CLAUDE_CSV}\n"
        f"PubChem CSV: {PUBCHEM_CSV}\n"
        f"Selection JSON: {FINAL_LIST_JSON}\n"
        f"Structures:\n"
        f"  Gemini: {STRUCT_GEMINI}\n"
        f"  OpenAI: {STRUCT_OPENAI}\n"
        f"  Claude: {STRUCT_CLAUDE}\n"
        f"  PubChem: {STRUCT_PUBCHEM}\n"
    )

    # Synkroniserer den aktuelle kørsel til projektets hovedmapper, så næste pipeline-trin kan bruge den seneste kørsel
    sync_latest_run_to_project_root()

    # Hvis RUN_COMPARISON_AFTER_GENERATION er sandt, starter sammenlignings- og alignments-pipelinen
    if RUN_COMPARISON_AFTER_GENERATION:
        print("\nStarting comparison + alignment pipeline ...")
        try:
            from Code_2_evaluation_pipeline import run_pipeline
            run_pipeline()
            print("\nFull pipeline complete ✅")
            print("Viewer should open automatically. If needed: python -m streamlit run Code_3_molecule_viewer.py")
        except Exception as e:
            print(f"\nPost-processing failed: {e}")
            raise


if __name__ == "__main__":
    main()