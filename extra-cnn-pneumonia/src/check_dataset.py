"""Validate the expected Kaggle dataset layout and image counts."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent / "data" / "chest_xray"
CONTAGENS_ESPERADAS = {
    "train": {"NORMAL": 1341, "PNEUMONIA": 3875},
    "val": {"NORMAL": 8, "PNEUMONIA": 8},
    "test": {"NORMAL": 234, "PNEUMONIA": 390},
}
EXTENSOES = {".jpeg", ".jpg", ".png"}


def contar(conjunto, classe):
    pasta = RAIZ / conjunto / classe
    if not pasta.is_dir():
        return None
    return sum(
        1
        for arquivo in pasta.iterdir()
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES
    )


def main():
    if not RAIZ.exists():
        print(f"Dataset não encontrado em {RAIZ}")
        print("Baixe e extraia o dataset conforme extra-cnn-pneumonia/README.md.")
        return 1

    print(f"{'conjunto':10s} {'NORMAL':>8s} {'PNEUMONIA':>10s} {'total':>7s}")
    ok = True
    for conjunto, esperadas in CONTAGENS_ESPERADAS.items():
        observadas = {}
        for classe, esperado in esperadas.items():
            quantidade = contar(conjunto, classe)
            observadas[classe] = quantidade
            if quantidade is None:
                print(f"{conjunto:10s} pasta {classe} faltando")
                ok = False
            elif quantidade != esperado:
                print(
                    f"{conjunto:10s} {classe}: encontradas {quantidade}, "
                    f"esperadas {esperado}"
                )
                ok = False

        if all(quantidade is not None for quantidade in observadas.values()):
            normal = observadas["NORMAL"]
            pneumonia = observadas["PNEUMONIA"]
            print(
                f"{conjunto:10s} {normal:8d} {pneumonia:10d} "
                f"{normal + pneumonia:7d}"
            )

    if not ok:
        print(
            "Dataset ausente, incompleto ou diferente da versão esperada. "
            "Confira a estrutura e as contagens no README."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
