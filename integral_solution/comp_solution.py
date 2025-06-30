import numpy as np
from scipy.integrate import quad
from scipy.special import i0e # IMPORTANTE: Trocamos i0 por i0e para estabilidade numérica
import matplotlib
matplotlib.use('Agg') # Define o backend não-interativo explicitamente
import matplotlib.pyplot as plt

def calcular_concentracao(c_0, D, a, r, t):
    """
    Calcula a concentração c(r,t) de forma numericamente estável.
    """
    if t <= 0:
        raise ValueError("O tempo (t) deve ser um número positivo.")
    if D <= 0 or a <= 0 or c_0 < 0 or r < 0:
        raise ValueError("D e 'a' devem ser positivos. 'c₀' e 'r' não podem ser negativos.")

    def integrando(r_prime, r_arg, D_arg, t_arg):
        # Argumento da função Bessel
        bessel_arg = (r_arg * r_prime) / (2 * D_arg * t_arg)
        
        # Usamos i0e(x) que calcula exp(-x)*i0(x) para evitar 'inf'.
        # Para compensar, multiplicamos o resultado por exp(bessel_arg).
        termo_bessel_estavel = i0e(bessel_arg)

        # O termo exp(-r'²/4Dt) é combinado com o exp(bessel_arg) da compensação.
        # exp(-(r'²)/(4Dt)) * exp((rr')/(2Dt)) = exp(-(r'² - 2rr')/(4Dt))
        expoente = (-(r_prime**2) + 2 * r_arg * r_prime) / (4 * D_arg * t_arg)
        termo_exp_combinado = np.exp(expoente)
        
        return termo_exp_combinado * termo_bessel_estavel * r_prime

    integral_result, _ = quad(integrando, 0, a, args=(r, D, t))
    
    # O fator externo permanece o mesmo
    fator_externo = (c_0 / (2 * D * t)) * np.exp(-(r**2) / (4 * D * t))
    
    concentracao_final = fator_externo * integral_result
    
    return concentracao_final

# --- Bloco principal para geração do gráfico ---
if __name__ == "__main__":
    
    # --- PARÂMETROS DA SIMULAÇÃO ---
    C_0 = 1.0
    
    # Solicitar D e A ao usuário
    while True:
        try:
            T = float(input("Digite o valor de T em °C (20): "))
            D = np.exp(-3.1e4 / (8.314 * (T + 273.15))) * 6e2  # cm²/s, usando a equação de Arrhenius
            if T <= -273.15:
                print("Temperatura deve ser um valor acima do zero absoluto. Tente novamente.")
            else:
                break
        except ValueError:
            print("Entrada inválida. Por favor, digite um número para T.")
        

    while True:
        try:
            A = float(input("Digite o valor de A em cm (ex: 0.5): "))
            if A <= 0:
                print("A deve ser um valor positivo. Tente novamente.")
            else:
                break
        except ValueError:
            print("Entrada inválida. Por favor, digite um número para A.")
    
    # --- CONFIGURAÇÕES DO GRÁFICO ---
    tempos_para_plotar = [0.1, 5, 10, 20, 50, 180]
    r_valores = np.linspace(0, 4, 200)

    # --- CÁLCULO E PLOTAGEM ---
    print("\nIniciando cálculo para a geração dos gráficos...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for t_atual in tempos_para_plotar:
        print(f"Calculando perfil de concentração para t = {t_atual}s...")
        concentracoes_calculadas = [calcular_concentracao(C_0, D, A, r, t_atual) for r in r_valores]
        ax.plot(r_valores, concentracoes_calculadas, label=f't = {t_atual} s')

    # --- CUSTOMIZAÇÃO DO GRÁFICO ---
    ax.set_title(f'Perfil de Concentração ao Longo do Tempo (D={D} cm²/s, a={A} cm)', fontsize=16)
    ax.set_xlabel('Distância Radial r (cm)', fontsize=12)
    ax.set_ylabel('Concentração Normalizada (c/c₀)', fontsize=12)
    ax.axvline(x=A, color='red', linestyle='--', label=f'Raio Inicial (a = {A})')
    ax.set_xlim(0, max(r_valores))
    ax.set_ylim(0, 1.05)
    ax.legend(title='Tempo (s)')
    plt.tight_layout()
    
    # --- SALVAR O GRÁFICO EM UM ARQUIVO ---
    nome_arquivo = 'concentracao_grafico.png'
    plt.savefig(nome_arquivo)
    
    print("\nCálculo finalizado com sucesso!")
    print(f"O gráfico não foi exibido na tela, mas foi salvo no arquivo: '{nome_arquivo}'")
