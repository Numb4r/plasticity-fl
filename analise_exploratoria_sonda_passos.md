# Calibração exploratória dos passos da sonda (proposta para a próxima rodada)

20 alvos (sementes >= 2000, fora das de avaliação), rede nova, protocolo da sonda (Adam lr 0.001, lotes de 100, MSE), perda nos 1000 pontos a cada passo.

- perda inicial (passo 0): mediana 0.025 [mín 0.0167, máx 0.041]
- perda no passo 300: mediana 0.000651; razão para a inicial: mediana 0.026 [mín 0.018, máx 0.034]
- primeiro passo com perda <= 30% da inicial: mediana 9 [mín 4, máx 11] (n=20)
- primeiro passo com perda <= 20% da inicial: mediana 14 [mín 12, máx 22] (n=20)
- primeiro passo com perda <= 10% da inicial: mediana 46 [mín 27, máx 78] (n=20)

**Leitura:** a rede nova já está entre 30% e 10% da perda inicial entre os passos ~9 e ~46; os 300 passos atuais levam a perda a 2.6% da inicial, ou seja, a sonda **satura** (o denominador deixa de discriminar). **Proposta:** usar um número de passos dentro dessa faixa, por exemplo 20 (média geométrica, arredondada a 5), e validar com redes treinadas que a cópia também não satura nem fica longe do alvo nesse número de passos.
