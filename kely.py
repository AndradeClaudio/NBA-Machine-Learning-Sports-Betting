def kelly_criterion(odds, probability, capital):
    b = odds - 1
    q = 1 - probability
    f = (b * probability - q) / b
    bet_amount = capital * f
    return bet_amount

def main():
    # Recebe odds, probabilidade e capital do usuário
    odds = float(input("Informe a odd da aposta: "))
    probability = float(input("Informe a probabilidade estimada (0 a 1): "))
    capital = float(input("Informe o seu capital atual: "))

    # Calcula o valor da aposta com base no Critério de Kelly
    bet_amount = kelly_criterion(odds, probability, capital)
    print(f"Valor sugerido para a aposta de acordo com o Critério de Kelly: R${bet_amount:.2f}")

if __name__ == "__main__":
    main()