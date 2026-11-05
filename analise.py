
# analise.py

from derivadas import (
    derivada_progressiva,
    derivada_regressiva,
    derivada_central
)


# Funções auxiliares

def numero_finito(valor):
    return (
        isinstance(valor, (int, float))
        and -float("inf") < valor < float("inf")
    )


def derivada_segura(f, x, h):
    try:
        y = f(x)
        d = derivada_central(f, x, h)

        if numero_finito(y) and numero_finito(d):
            return d

    except (ZeroDivisionError, ValueError, OverflowError):
        pass

    return None


# 1. Analisar convergência

def analisar_convergencia(resultados, epsilon, metodo="progressiva"):
    anterior = None
    convergiu = False

    for item in resultados:
        atual = item[metodo]

        if anterior is not None:
            diferenca = abs(atual - anterior)

            print(f"h = {item['h']}")
            print(f"Diferença = {diferenca:.8f}")

            convergiu = diferenca < epsilon

        anterior = atual

    return convergiu


# 2. Verificar derivabilidade

def verificar_derivabilidade(f, x, h, epsilon):
    if h <= 0 or epsilon <= 0:
        raise ValueError("h e epsilon devem ser positivos")

    try:
        if not numero_finito(f(x)):
            return "Inconclusivo"
    except (ZeroDivisionError, ValueError, OverflowError):
        return "Inconclusivo"

    anterior_progressiva = None
    anterior_regressiva = None

    passo = h
    variacao = float("inf")
    diferenca = float("inf")

    for k in range(12):
        if x + passo == x or x - passo == x:
            break

        try:
            progressiva = derivada_progressiva(f, x, passo)
            regressiva = derivada_regressiva(f, x, passo)

        except (ValueError, ZeroDivisionError, OverflowError):
            return "Inconclusivo"

        if not (
            numero_finito(progressiva)
            and numero_finito(regressiva)
        ):
            return "Inconclusivo"

        diferenca = abs(progressiva - regressiva)

        if anterior_progressiva is not None:
            variacao = max(
                abs(progressiva - anterior_progressiva),
                abs(regressiva - anterior_regressiva)
            )

            if diferenca < epsilon and variacao < epsilon:
                return "Derivável (aproximadamente)"

        anterior_progressiva = progressiva
        anterior_regressiva = regressiva

        passo /= 2

    if diferenca >= epsilon and variacao < epsilon:
        return "Possivelmente não derivável"

    return "Inconclusivo"


# 3. Analisar sinal

def analisar_sinal(f, valores_x, h, epsilon):
    resultados = []

    for x in valores_x:
        derivada = derivada_segura(f, x, h)

        if derivada is None:
            sinal = "Fora do domínio ou derivada indisponível"

        elif derivada > epsilon:
            sinal = "Crescente"

        elif derivada < -epsilon:
            sinal = "Decrescente"

        else:
            situacao = verificar_derivabilidade(
                f, x, h, epsilon
            )

            if situacao == "Possivelmente não derivável":
                sinal = "Derivada possivelmente inexistente"

            elif situacao == "Derivável (aproximadamente)":
                sinal = "Derivada próxima de zero"

            else:
                sinal = "Inconclusivo"

        resultados.append({
            "x": x,
            "derivada": derivada,
            "sinal": sinal
        })

    return resultados


# 4. Encontrar pontos críticos

def encontrar_pontos_criticos(f, valores_x, h, epsilon):
    if h <= 0 or epsilon <= 0:
        raise ValueError("h e epsilon devem ser positivos")

    pontos = []
    valores_x = sorted(set(valores_x))

    def adicionar_ponto(x):
        derivada = derivada_segura(f, x, h)

        if derivada is None or abs(derivada) >= epsilon:
            return

        for ponto in pontos:
            if abs(ponto["x"] - x) < epsilon:
                return

        pontos.append({
            "x": x,
            "f(x)": f(x),
            "derivada": derivada
        })

    # Calcular as derivadas nos pontos conhecidos
    derivadas = []

    for x in valores_x:
        d = derivada_segura(f, x, h)
        derivadas.append(d)

        if d is not None and abs(d) < epsilon:
            adicionar_ponto(x)

    # Procurar zeros por mudança de sinal
    for i in range(len(valores_x) - 1):
        a = valores_x[i]
        b = valores_x[i + 1]

        da = derivadas[i]
        db = derivadas[i + 1]

        if da is None or db is None:
            continue

        if da * db < 0:
            for k in range(80):
                meio = (a + b) / 2
                dm = derivada_segura(f, meio, h)

                if dm is None:
                    break

                if abs(dm) < epsilon:
                    adicionar_ponto(meio)
                    break

                if da * dm < 0:
                    b = meio
                else:
                    a = meio
                    da = dm

    # Procurar mínimos locais do módulo da derivada
    # Isso também ajuda a localizar zeros sem mudança de sinal
    for i in range(1, len(valores_x) - 1):
        da = derivadas[i - 1]
        dm = derivadas[i]
        db = derivadas[i + 1]

        if da is None or dm is None or db is None:
            continue

        if not (abs(dm) <= abs(da) and abs(dm) <= abs(db)):
            continue

        esquerda = valores_x[i - 1]
        direita = valores_x[i + 1]

        for k in range(40):
            p = esquerda + (direita - esquerda) / 3
            q = direita - (direita - esquerda) / 3

            dp = derivada_segura(f, p, h)
            dq = derivada_segura(f, q, h)

            if dp is None or dq is None:
                break

            if abs(dp) < abs(dq):
                direita = q
            else:
                esquerda = p

        meio = (esquerda + direita) / 2
        adicionar_ponto(meio)

    return sorted(
        pontos,
        key=lambda ponto: ponto["x"]
    )


# 5. Classificar pontos críticos

def classificar_pontos_criticos(
    f, pontos, h, epsilon, distancia=0.1
):
    resultados = []

    for ponto in pontos:
        x = ponto["x"]

        situacao = verificar_derivabilidade(
            f, x, h, epsilon
        )

        esquerda = derivada_segura(
            f, x - distancia, h
        )

        direita = derivada_segura(
            f, x + distancia, h
        )

        if esquerda is None or direita is None:
            classificacao = "Inconclusivo"

        elif esquerda < -epsilon and direita > epsilon:
            classificacao = "Mínimo local"

        elif esquerda > epsilon and direita < -epsilon:
            classificacao = "Máximo local"

        elif (
            esquerda > epsilon and direita > epsilon
        ) or (
            esquerda < -epsilon and direita < -epsilon
        ):
            classificacao = "Não é extremo local"

        else:
            classificacao = "Inconclusivo"

        resultados.append({
            "x": x,
            "f(x)": ponto["f(x)"],
            "derivabilidade": situacao,
            "classificacao": classificacao
        })

    return resultados
