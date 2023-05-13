def kelly_criterion(odds, probability, capital):
    b = odds - 1
    q = 1 - probability
    f = (b * probability - q) / b
    bet_amount = capital * f
    return bet_amount

def calculate_potential_profit(odds, bet_amount):
    profit = bet_amount * (odds - 1)
    return profit

def main():
    # Recebe odds, probabilidade e capital do usuário
    odds = float(input("Informe a odd da aposta: "))
    probability = float(input("Informe a probabilidade estimada (0 a 1): "))
    #capital = 9.44
    capital = 9.64
    # Calcula o valor da aposta com base no Critério de Kelly
    bet_amount = kelly_criterion(odds, probability, capital)
    if bet_amount>0:
        print(f"Valor sugerido para a aposta de acordo com o Critério de Kelly: R${bet_amount:.2f}")
        # Calcula o lucro potencial com base no valor da aposta recomendado
        potential_profit = calculate_potential_profit(odds, bet_amount)
        print(f"Lucro potencial se a aposta for bem-sucedida: R${potential_profit:.2f}")
    else:
        print("Não é recomendado apostar nesse jogo")
if __name__ == "__main__":
    main()
