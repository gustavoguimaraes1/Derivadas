
# derivadas.py


# 1. Derivada progressiva
def derivada_progressiva(f, x, h):
    if h <= 0:
        raise ValueError("O passo h deve ser positivo")

    return (f(x + h) - f(x)) / h


# 2. Derivada regressiva
def derivada_regressiva(f, x, h):
    if h <= 0:
        raise ValueError("O passo h deve ser positivo")

    return (f(x) - f(x - h)) / h


# 3. Derivada central
def derivada_central(f, x, h):
    if h <= 0:
        raise ValueError("O passo h deve ser positivo")

    return (f(x + h) - f(x - h)) / (2 * h)


# 4. Refinamento da derivada numérica
def derivada(f, x, h, epsilon, kmax, metodo="central"):

    metodos = {
        "progressiva": derivada_progressiva,
        "regressiva": derivada_regressiva,
        "central": derivada_central
    }

    # Verificar o método
    if metodo not in metodos:
        raise ValueError("Método inválido")

    # Verificar os parâmetros
    if h <= 0 or epsilon <= 0:
        raise ValueError("h e epsilon devem ser positivos")

    if not isinstance(kmax, int) or kmax < 1:
        raise ValueError("kmax deve ser um inteiro positivo")

    # Selecionar o método
    calcular = metodos[metodo]

    # Primeira aproximação
    anterior = calcular(f, x, h)

    # Refinamento iterativo
    for k in range(1, kmax + 1):
        novo_h = h / 2

        # Evitar passos pequenos demais
        if x + novo_h == x or x - novo_h == x:
            return anterior, k - 1, False

        h = novo_h

        # Calcular nova aproximação
        atual = calcular(f, x, h)

        # Diferença entre aproximações
        erro = abs(atual - anterior)

        # Verificar estabilização
        if erro < epsilon:
            return atual, k, True

        anterior = atual

    return anterior, kmax, False
