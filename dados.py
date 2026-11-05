
# dados.py

import pandas as pd

from derivadas import (
    derivada_progressiva,
    derivada_regressiva,
    derivada_central
)


# Funções auxiliares

def avaliar_seguro(f, x):
    try:
        resultado = f(x)

        if -float("inf") < resultado < float("inf"):
            return resultado

    except (ZeroDivisionError, ValueError, OverflowError):
        pass

    return float("nan")


def calcular_derivada_segura(metodo, f, x, h):
    # A função deve existir no próprio ponto
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


# 1. Testar diferentes valores de h

def testar_valores_h(f, x0, valores_h):
    resultados = []

    for h in valores_h:
        progressiva = calcular_derivada_segura(
            derivada_progressiva, f, x0, h
        )

        regressiva = calcular_derivada_segura(
            derivada_regressiva, f, x0, h
        )

        central = calcular_derivada_segura(
            derivada_central, f, x0, h
        )

        resultados.append({
            "h": h,
            "progressiva": progressiva,
            "regressiva": regressiva,
            "central": central
        })

    return resultados


# 2. Gerar valores de x

def gerar_valores_x(inicio, fim, passo):
    if passo <= 0:
        raise ValueError("O passo deve ser positivo")

    valores = []
    x = inicio

    while x <= fim:
        valores.append(round(x, 10))
        x += passo

    return valores


# 3. Criar DataFrame da função

def criar_dataframe_funcao(f, valores_x):
    dados = []

    for x in valores_x:
        dados.append({
            "x": x,
            "f(x)": avaliar_seguro(f, x)
        })

    return pd.DataFrame(
        dados,
        columns=["x", "f(x)"]
    )


# 4. Criar DataFrame das derivadas

def criar_dataframe_derivada(f, valores_x, h):
    dados = []

    for x in valores_x:
        dados.append({
            "x": x,
            "Progressiva": calcular_derivada_segura(
                derivada_progressiva, f, x, h
            ),
            "Regressiva": calcular_derivada_segura(
                derivada_regressiva, f, x, h
            ),
            "Central": calcular_derivada_segura(
                derivada_central, f, x, h
            )
        })

    return pd.DataFrame(
        dados,
        columns=[
            "x",
            "Progressiva",
            "Regressiva",
            "Central"
        ]
    )


# 5. Criar DataFrame completo

def criar_dataframe_completo(f, valores_x, h):
    tabela_funcao = criar_dataframe_funcao(
        f, valores_x
    )

    tabela_derivadas = criar_dataframe_derivada(
        f, valores_x, h
    )

    tabela_completa = pd.merge(
        tabela_funcao,
        tabela_derivadas,
        on="x"
    )

    return tabela_completa
