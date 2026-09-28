"""Confere se o dataset está na pasta certa e conta as imagens por conjunto e classe."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent / "data" / "chest_xray"
CONJUNTOS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]
EXTENSOES = {".jpeg", ".jpg", ".png"}


def contar(conjunto, classe):
    pasta = RAIZ / conjunto / classe
    return sum(1 for f in pasta.iterdir() if f.suffix.lower() in EXTENSOES) if pasta.exists() else None


def main():
    if not RAIZ.exists():
        print(f"Dataset não encontrado em {RAIZ}")
        print("Veja as instruções de download no README.md")
        return 1
    print(f"{'conjunto':10s} {'NORMAL':>8s} {'PNEUMONIA':>10s} {'total':>7s}")
    ok = True
    for conj in CONJUNTOS:
        qtd = [contar(conj, c) for c in CLASSES]
        if None in qtd:
            print(f"{conj:10s} pasta faltando")
            ok = False
            continue
        print(f"{conj:10s} {qtd[0]:8d} {qtd[1]:10d} {sum(qtd):7d}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
