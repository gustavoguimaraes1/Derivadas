
# newton.py

from derivadas import derivada_central


# 1. Método de Newton-Raphson
def metodo_newton(f, x0, h, epsilon, kmax, historico=None):

    # Verificar os parâmetros
    if h <= 0 or epsilon <= 0:
        raise ValueError("h e epsilon devem ser positivos")

    if not isinstance(kmax, int) or kmax < 1:
        raise ValueError("kmax deve ser um inteiro positivo")

    # Aproximação inicial
    x = x0

    if historico is not None:
        historico.append(x)

    # Processo iterativo
    for k in range(kmax):

        try:
            # Calcular o valor da função
            fx = f(x)

            if not -float("inf") < fx < float("inf"):
                return x, k, False

            # Verificar raiz exata
            if fx == 0:
                return x, k, True

            # Calcular derivada numérica
            dfx = derivada_central(f, x, h)

            if not -float("inf") < dfx < float("inf"):
                return x, k, False

            # Evitar divisão por derivada próxima de zero
            if abs(dfx) < 1e-12:
                return x, k, False

            # Fórmula de Newton-Raphson
            proximo = x - fx / dfx

            if not -float("inf") < proximo < float("inf"):
                return x, k, False

            # Avaliar a nova aproximação
            novo_fx = f(proximo)

            if not -float("inf") < novo_fx < float("inf"):
                return x, k, False

        except (ZeroDivisionError, ValueError, OverflowError):
            return x, k, False

        # Armazenar a aproximação
        if historico is not None:
            historico.append(proximo)

        # Calcular a variação entre aproximações
        erro = abs(proximo - x)

        tolerancia_passo = epsilon * max(
            1, abs(proximo)
        )

        # Verificar convergência
        if (
            abs(novo_fx) < epsilon
            and erro < tolerancia_passo
        ):
            return proximo, k + 1, True

        # Atualizar a aproximação
        x = proximo

    # Limite de iterações atingido
    return x, kmax, False
