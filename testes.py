# testes.py
# Execute: python -m unittest -v testes.py

import math
import unittest

from derivadas import derivada_central, derivada
from dados import gerar_valores_x, criar_dataframe_completo
from analise import encontrar_pontos_criticos, classificar_pontos_criticos
from newton import metodo_newton


# (nome, funcao, derivada exata em x=3, pontos esperados, raiz esperada)
# Um ponto esperado e uma tupla (abscissa, classificacao).
# None na raiz significa que Newton nao precisa convergir naquele teste.
CASOS = [
    ("quadratica", lambda x: x**2 - 6, 6.0,
     [(0.0, "Mínimo local")], math.sqrt(6)),
    ("cubica_com_tres_zeros", lambda x: x**3 - 4*x, 23.0,
     [(-math.sqrt(4/3), "Máximo local"),
      (math.sqrt(4/3), "Mínimo local")], 2.0),
    ("quartica_menos_4x", lambda x: x**4 - 4*x, 104.0,
     [(1.0, "Mínimo local")], 4**(1/3)),
    ("quartica_com_tres_criticos", lambda x: x**4 - 2*x**2, 96.0,
     [(-1.0, "Mínimo local"), (0.0, "Máximo local"),
      (1.0, "Mínimo local")], math.sqrt(2)),
    ("cubica_ponto_estacionario", lambda x: x**3, 27.0,
     [(0.0, "Não é extremo local")], 0.0),
    ("modulo", lambda x: abs(x), 1.0,
     [(0.0, "Mínimo local")], None),
    ("racional_sem_raiz", lambda x: 1/(x + 2), -0.04,
     [], None),
    ("cubica_deslocada", lambda x: (x - 0.37)**3, 3*(3 - 0.37)**2,
     [(0.37, "Não é extremo local")], 0.37),
    ("cubica_com_maximo_minimo", lambda x: x**3 - 3*x + 1, 24.0,
     [(-1.0, "Máximo local"), (1.0, "Mínimo local")],
     1.532088886237956),
    ("racional_com_furo", lambda x: (x**2 - 1)/(x - 1), 1.0,
     [], -1.0),
    ("modulo_deslocado", lambda x: abs(x - 1), 1.0,
     [(1.0, "Mínimo local")], None),
]


class TestesDerivadasNumericas(unittest.TestCase):
    # Um teste para cada uma das 11 funcoes matematicas.
    def verificar_caso(self, nome, f, derivada_exata, esperados, raiz_esperada):
        # 1. Comparar a derivada central com o valor exato em x=3.
        aproximacao = derivada_central(f, 3, 0.0001)
        self.assertAlmostEqual(
            aproximacao, derivada_exata, delta=0.001,
            msg=f"Derivada incorreta: {nome}"
        )

        # 2. Localizar e classificar os pontos criticos entre -3 e 3.
        valores_x = gerar_valores_x(-3, 3, 0.1)
        pontos = encontrar_pontos_criticos(f, valores_x, 0.001, 0.0001)
        classificados = classificar_pontos_criticos(f, pontos, 0.001, 0.0001)

        self.assertEqual(
            len(classificados), len(esperados),
            msg=f"Pontos criticos inesperados em {nome}: {classificados}"
        )

        for x_esperado, classe_esperada in esperados:
            encontrados = [
                ponto for ponto in classificados
                if abs(ponto["x"] - x_esperado) < 0.01
            ]
            self.assertEqual(
                len(encontrados), 1,
                msg=f"Ponto x={x_esperado} nao encontrado: {nome}"
            )
            self.assertEqual(encontrados[0]["classificacao"], classe_esperada)

        # 3. Testar Newton-Raphson com as mesmas configuracoes do main.py.
        historico = []
        x, iteracoes, convergiu = metodo_newton(
            f, 3, 0.001, 0.0001, 100, historico
        )

        self.assertGreaterEqual(iteracoes, 0)
        self.assertLessEqual(iteracoes, 100)
        self.assertTrue(historico)

        if nome == "racional_sem_raiz":
            self.assertFalse(convergiu, "A funcao 1/(x+2) nao possui raiz")
        elif raiz_esperada is not None:
            self.assertTrue(convergiu, f"Newton nao convergiu: {nome}")
            self.assertAlmostEqual(x, raiz_esperada, delta=0.01)
            self.assertLess(abs(f(x)), 0.0001)
        elif convergiu:
            # Funcoes modulo nao precisam convergir, mas uma convergencia
            # relatada deve ter residuo pequeno.
            self.assertLess(abs(f(x)), 0.0001)

    def test_01_quadratica(self):
        self.verificar_caso(*CASOS[0])

    def test_02_cubica_com_tres_zeros(self):
        self.verificar_caso(*CASOS[1])

    def test_03_quartica_menos_4x(self):
        self.verificar_caso(*CASOS[2])

    def test_04_quartica_com_tres_criticos(self):
        self.verificar_caso(*CASOS[3])

    def test_05_cubica_ponto_estacionario(self):
        self.verificar_caso(*CASOS[4])

    def test_06_modulo(self):
        self.verificar_caso(*CASOS[5])

    def test_07_racional_sem_raiz(self):
        self.verificar_caso(*CASOS[6])

    def test_08_cubica_deslocada(self):
        self.verificar_caso(*CASOS[7])

    def test_09_cubica_com_maximo_minimo(self):
        self.verificar_caso(*CASOS[8])

    def test_10_racional_com_furo(self):
        self.verificar_caso(*CASOS[9])

    def test_11_modulo_deslocado(self):
        self.verificar_caso(*CASOS[10])

    # Testes complementares de integracao dos modulos.
    def test_12_dataframe_com_furo(self):
        f = lambda x: (x**2 - 1)/(x - 1)
        tabela = criar_dataframe_completo(f, [0, 1, 2, 3], 0.1)

        self.assertTrue(math.isnan(tabela.loc[1, "f(x)"]))
        for coluna in ("Progressiva", "Regressiva", "Central"):
            self.assertTrue(math.isnan(tabela.loc[1, coluna]))
        self.assertAlmostEqual(tabela.loc[2, "Central"], 1.0, delta=1e-5)

    def test_13_refinamento_derivada(self):
        valor, iteracoes, convergiu = derivada(
            lambda x: x**2 - 6, 3, 0.1, 0.0001, 100, "progressiva"
        )
        self.assertTrue(convergiu)
        self.assertAlmostEqual(valor, 6, delta=0.001)
        self.assertGreater(iteracoes, 0)

    def test_14_derivabilidade_modulo(self):
        from analise import verificar_derivabilidade
        situacao = verificar_derivabilidade(abs, 0, 0.001, 0.0001)
        self.assertEqual(situacao, "Possivelmente não derivável")

    def test_15_sinais_no_furo(self):
        from analise import analisar_sinal
        f = lambda x: (x**2 - 1)/(x - 1)
        resultados = analisar_sinal(f, [0, 1, 2], 0.001, 0.0001)
        self.assertEqual(resultados[1]["derivada"], None)
        self.assertEqual(resultados[0]["sinal"], "Crescente")
        self.assertEqual(resultados[2]["sinal"], "Crescente")

