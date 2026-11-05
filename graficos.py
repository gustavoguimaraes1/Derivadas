# graficos.py

import matplotlib.pyplot as plt
from pathlib import Path

from derivadas import (
    derivada_progressiva,
    derivada_regressiva,
    derivada_central
)


# Funções auxiliares

def salvar_grafico(nome):
    # Criar a pasta de imagens
    pasta = Path(__file__).resolve().parent / "imagens"
    pasta.mkdir(exist_ok=True)

    # Salvar o gráfico em PNG
    plt.savefig(
        pasta / f"{nome}.png",
        dpi=300,
        bbox_inches="tight"
    )

    # Exibir o gráfico
    plt.show()
    plt.close()


def avaliar_seguro(f, x):
    try:
        resultado = f(x)

        if -float("inf") < resultado < float("inf"):
            return resultado

    except (ZeroDivisionError, ValueError, OverflowError):
        pass

    return float("nan")


def calcular_derivada_segura(metodo, f, x, h):
    # Verificar se a função existe no ponto
    valor = avaliar_seguro(f, x)

    if not -float("inf") < valor < float("inf"):
        return float("nan")

    try:
        resultado = metodo(f, x, h)

        if -float("inf") < resultado < float("inf"):
            return resultado

    except (ZeroDivisionError, ValueError, OverflowError):
        pass

    return float("nan")


# 1. Gráfico da função

def grafico_funcao(f, valores_x):
    valores_y = []

    for x in valores_x:
        valores_y.append(avaliar_seguro(f, x))

    plt.figure(figsize=(8, 5))

    plt.plot(
        valores_x,
        valores_y,
        label="f(x)",
        color="blue"
    )

    plt.title("Gráfico da Função")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    salvar_grafico("funcao")


# 2. Gráfico das derivadas numéricas

def grafico_derivada(f, valores_x, h):
    if h <= 0:
        raise ValueError("O passo h deve ser positivo")

    progressiva = []
    regressiva = []
    central = []

    for x in valores_x:
        progressiva.append(
            calcular_derivada_segura(
                derivada_progressiva, f, x, h
            )
        )

        regressiva.append(
            calcular_derivada_segura(
                derivada_regressiva, f, x, h
            )
        )

        central.append(
            calcular_derivada_segura(
                derivada_central, f, x, h
            )
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        valores_x,
        progressiva,
        label="Progressiva"
    )

    plt.plot(
        valores_x,
        regressiva,
        label="Regressiva"
    )

    plt.plot(
        valores_x,
        central,
        label="Central"
    )

    plt.title("Comparação das Derivadas Numéricas")
    plt.xlabel("x")
    plt.ylabel("f'(x)")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    salvar_grafico("derivadas")


# 3. Gráfico de convergência

def grafico_convergencia(resultados):
    valores_h = []
    progressiva = []
    regressiva = []
    central = []

    for item in resultados:
        valores_h.append(item["h"])
        progressiva.append(item["progressiva"])
        regressiva.append(item["regressiva"])
        central.append(item["central"])

    if not valores_h or any(h <= 0 for h in valores_h):
        raise ValueError(
            "A convergência exige passos h positivos"
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        valores_h,
        progressiva,
        marker="o",
        label="Progressiva"
    )

    plt.plot(
        valores_h,
        regressiva,
        marker="s",
        label="Regressiva"
    )

    plt.plot(
        valores_h,
        central,
        marker="^",
        label="Central"
    )

    plt.xscale("log")
    plt.gca().invert_xaxis()

    plt.title("Convergência das Derivadas Numéricas")
    plt.xlabel("Passo h")
    plt.ylabel("Aproximação da derivada")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    salvar_grafico("convergencia")


# 4. Gráfico de Newton-Raphson

def grafico_newton(f, valores_x, historico):
    valores_y = []

    for x in valores_x:
        valores_y.append(avaliar_seguro(f, x))

    plt.figure(figsize=(8, 5))

    plt.plot(
        valores_x,
        valores_y,
        label="f(x)"
    )

    plt.axhline(
        0,
        linestyle="--",
        color="gray"
    )

    for k in range(len(historico) - 1):
        x_atual = historico[k]
        x_proximo = historico[k + 1]

        y_atual = avaliar_seguro(f, x_atual)
        y_proximo = avaliar_seguro(f, x_proximo)

        if not (
            -float("inf") < y_atual < float("inf")
            and -float("inf") < y_proximo < float("inf")
        ):
            continue

        plt.scatter(x_atual, y_atual)

        # Representar a reta tangente aproximada
        plt.plot(
            [x_atual, x_proximo],
            [y_atual, 0],
            linestyle="--"
        )

        # Representar a próxima aproximação
        plt.plot(
            [x_proximo, x_proximo],
            [0, y_proximo],
            linestyle=":"
        )

    # Destacar a última aproximação
    if historico:
        x_final = historico[-1]
        y_final = avaliar_seguro(f, x_final)

        if -float("inf") < y_final < float("inf"):
            plt.scatter(
                x_final,
                y_final,
                label="Aproximação final"
            )

    plt.title("Método de Newton-Raphson")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    salvar_grafico("newton")


# 5. Marcar máximos e mínimos locais

def marcar_maximos_minimos(f, valores_x, classificados):
    valores_y = []

    for x in valores_x:
        valores_y.append(avaliar_seguro(f, x))

    plt.figure(figsize=(8, 5))

    plt.plot(
        valores_x,
        valores_y,
        label="f(x)"
    )

    for ponto in classificados:
        x = ponto["x"]
        y = ponto["f(x)"]
        classificacao = ponto["classificacao"]

        if classificacao == "Mínimo local":
            plt.scatter(
                x,
                y,
                color="green",
                s=100
            )

            plt.annotate(
                "Mínimo local",
                (x, y),
                xytext=(8, -15),
                textcoords="offset points"
            )

        elif classificacao == "Máximo local":
            plt.scatter(
                x,
                y,
                color="red",
                s=100
            )

            plt.annotate(
                "Máximo local",
                (x, y),
                xytext=(8, 10),
                textcoords="offset points"
            )

    plt.title("Máximos e Mínimos Locais")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    salvar_grafico("maximos_minimos")
