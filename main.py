
# main.py

from funcao import funcao
from derivadas import derivada

from dados import (
    testar_valores_h,
    gerar_valores_x,
    criar_dataframe_funcao,
    criar_dataframe_derivada,
    criar_dataframe_completo
)

from newton import metodo_newton

from analise import (
    analisar_convergencia,
    analisar_sinal,
    encontrar_pontos_criticos,
    classificar_pontos_criticos
)

from graficos import (
    grafico_funcao,
    grafico_derivada,
    grafico_convergencia,
    grafico_newton,
    marcar_maximos_minimos
)


def main():

    # 1. Definição dos parâmetros
    x0 = 3

    valores_h = [0.1, 0.01, 0.001, 0.0001]

    epsilon = 0.0001
    kmax = 100

    h_inicial = 0.1
    h_analise = 0.001
    h_newton = 0.001

    metodo = "progressiva"

    # 2. Testes das derivadas numéricas
    resultados = testar_valores_h(
        funcao, x0, valores_h
    )

    print("\nTestes com diferentes valores de h:")

    for item in resultados:
        print(item)

    # 3. Geração dos valores de x
    valores_x = gerar_valores_x(0, 5, 1)

    # 4. Tabela da função
    tabela = criar_dataframe_funcao(
        funcao, valores_x
    )

    print("\nTabela da função:")
    print(tabela)

    # 5. Tabela das derivadas
    tabela_derivadas = criar_dataframe_derivada(
        funcao,
        valores_x,
        h_inicial
    )

    print("\nTabela das derivadas:")
    print(tabela_derivadas)

    # 6. Tabela completa
    tabela_completa = criar_dataframe_completo(
        funcao,
        valores_x,
        h_inicial
    )

    print("\nTabela completa:")
    print(
        tabela_completa.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.4f}"
        )
    )

    # 7. Refinamento da derivada numérica
    resultado, iteracoes, convergiu = derivada(
        funcao,
        x0,
        h_inicial,
        epsilon,
        kmax,
        metodo
    )

    print("\nRefinamento da derivada:")
    print(f"Resultado: {resultado:.6f}")
    print(f"Iterações: {iteracoes}")
    print(f"Convergência: {convergiu}")

    # 8. Método de Newton-Raphson
    historico_newton = []

    raiz, iteracoes_newton, convergiu_newton = metodo_newton(
        funcao,
        x0,
        h_newton,
        epsilon,
        kmax,
        historico_newton
    )

    print("\nMétodo de Newton-Raphson:")

    if convergiu_newton:
        print(f"Raiz aproximada: {raiz:.6f}")

    else:
        print("Newton-Raphson não convergiu.")
        print(f"Última aproximação: {raiz:.6f}")

    print(f"Iterações: {iteracoes_newton}")
    print(f"Convergência: {convergiu_newton}")

    # 9. Análise de convergência
    print("\nAnálise de convergência:")

    convergiu_h = analisar_convergencia(
        resultados,
        epsilon,
        metodo
    )

    print(f"Estabilizou: {convergiu_h}")

    # 10. Análise dos sinais
    sinais = analisar_sinal(
        funcao,
        valores_x,
        h_analise,
        epsilon
    )

    print("\nAnálise dos sinais:")

    for item in sinais:
        print(item)

    # 11. Identificação dos pontos críticos
    valores_analise = gerar_valores_x(
        -3, 3, 0.1
    )

    pontos = encontrar_pontos_criticos(
        funcao,
        valores_analise,
        h_analise,
        epsilon
    )

    print("\nPontos críticos encontrados:")

    if pontos:
        for ponto in pontos:
            print(ponto)
    else:
        print("Nenhum ponto crítico encontrado.")

    # 12. Classificação dos pontos críticos
    classificados = classificar_pontos_criticos(
        funcao,
        pontos,
        h_analise,
        epsilon
    )

    print("\nClassificação dos pontos críticos:")

    if classificados:
        for item in classificados:
            print(item)
    else:
        print("Nenhum ponto crítico para classificar.")

    # 13. Geração dos gráficos
    valores_grafico = gerar_valores_x(
        0, 5, 0.1
    )

    grafico_funcao(
        funcao,
        valores_grafico
    )

    grafico_derivada(
        funcao,
        valores_grafico,
        h_inicial
    )

    grafico_convergencia(
        resultados
    )

    if convergiu_newton:
        grafico_newton(
            funcao,
            valores_grafico,
            historico_newton
        )

    marcar_maximos_minimos(
        funcao,
        valores_analise,
        classificados
    )


if __name__ == "__main__":
    main()
