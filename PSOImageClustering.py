import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

slika = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
slika = cv2.resize(slika, (256, 256))


def prosjecneUdaljenostiPoKlasterima(x, slika):
    klasterColors = DajBojeZaKlastere(x)
    hist = cv2.calcHist([slika], [0], None, [256], [0, 256])
    udaljenosti = []
    for i in range(len(x)):
        brojPiksela = 0
        suma = 0
        for boja in klasterColors[i]:
            brojPiksela += hist[boja]
            suma += hist[boja] * abs(boja - x[i])
        if brojPiksela > 0:
            udaljenosti.append(suma / brojPiksela)
        else:
            # prazan klaseter se "kaznjava" sa vellikom distancom
            udaljenosti.append(10000)
    return udaljenosti


def prosjecnaUdaljenost(x, slika):
    udaljenosti = prosjecneUdaljenostiPoKlasterima(x, slika)
    return sum(udaljenosti) / len(x)


def DajBojeZaKlastere(x):
    klasterColors = []
    for i in range(len(x)):
        klasterColors.append([])
    for boja in range(0, 256):
        najBliziKlaster = 0
        for i in range(len(x)):
            if abs(boja - x[i]) < abs(boja - x[najBliziKlaster]):
                najBliziKlaster = i
        klasterColors[najBliziKlaster].append(boja)
    return klasterColors


def minimalnaUdaljenostIzmedjuKlastera(x):
    minDist = 0
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j and abs(x[i] - x[j]) < minDist:
                minDist = abs(x[i] - x[j])
    return minDist


def maksimalnaUdaljenost(x, slika):
    udaljenosti = prosjecneUdaljenostiPoKlasterima(x, slika)
    return max(udaljenosti)


def funkcijaDistance1(x, slika):
    return 5 * maksimalnaUdaljenost(x, slika) + 0.1 * (255 - minimalnaUdaljenostIzmedjuKlastera(x))


def sumaUdaljenosti(x, slika):
    udaljenosti = prosjecneUdaljenostiPoKlasterima(x, slika)
    return sum(udaljenosti)


def udaljenostiKlastera(x):
    udaljenost = 0
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                udaljenost += abs(x[i] - x[j]) ** 2
    return udaljenost


def funkcijaDistance2(x, slika):
    alfa = SSE(x, slika)
    beta = udaljenostiKlastera(x)
    return alfa / beta


def SSE(x, slika):
    klasterColors = DajBojeZaKlastere(x)
    hist = cv2.calcHist([slika], [0], None, [256], [0, 256])
    SSE = 0
    for i in range(len(x)):
        brojPiksela = 0
        for boja in klasterColors[i]:
            brojPiksela += hist[boja]
            SSE += hist[boja] * abs(boja - x[i]) ** 2
        # Prazan klaster se kaznnjava sa velikim SSE
        if brojPiksela == 0:
            SSE += 1000000.
    return SSE[0]


def funkcijaDistance3(x, slika):
    return SSE(x, slika)


def PSO(f, slika, brojCestica, brojIteracija, brojKlastera, P):
    x = []
    v = []
    omega = P[0]
    c1 = P[1]
    c2 = P[2]
    for i in range(0, brojCestica):
        x.append(np.random.uniform(slika.min(), slika.max(), brojKlastera))
        v.append(np.random.uniform(0, 5, brojKlastera))
    pb = x
    min = x[0]
    for i in range(1, brojCestica):
        if f(x[i], slika) < f(min, slika):
            min = x[i]
    gb = min
    iter = 0
    SSEErr = []
    avgDist = []
    while iter < brojIteracija:
        for i in range(0, brojCestica):
            r1, r2 = np.random.uniform(0, 1.0000000001, 2)
            v_novo = omega * v[i] + c1 * r1 * (pb[i] - x[i]) + c2 * r2 * (gb - x[i])
            x_novo = x[i] + v_novo
            veciOdOpsega = x_novo[x_novo > slika.max()]
            manjiOdOpsega = x_novo[x_novo < slika.min()]
            if len(veciOdOpsega) > 0 or len(manjiOdOpsega) > 0:
                continue
            f_x = f(x_novo, slika)
            if f_x < f(pb[i], slika):
                pb[i] = np.copy(x_novo)
            if f_x < f(gb, slika):
                gb = np.copy(x_novo)
            x[i] = x_novo
            v[i] = v_novo
        SSEErr.append(SSE(gb, slika))
        avgDist.append(prosjecnaUdaljenost(gb, slika))
        iter += 1
    return {
        "rez": gb,
        "SSE": SSEErr,
        "avgDist": avgDist
    }


def PSO2(f, slika, brojCestica, brojIteracija, brojKlastera, P):
    x = []
    v = []
    omega = P[0]
    c1 = P[1]
    c2 = P[2]
    for i in range(0, brojCestica):
        x.append(np.random.uniform(slika.min(), slika.max(), brojKlastera))
        v.append(np.random.uniform(0, 5, brojKlastera))
    pb = x
    min = x[0]
    for i in range(1, brojCestica):
        if f(x[i], slika) < f(min, slika):
            min = x[i]
    gb = min
    iter = 0
    SSEErr = []
    avgDist = []

    while iter < brojIteracija:
        omega = 0.95 - iter * (0.95 - 0.5) / brojIteracija
        c1 = (P[1] * np.e ** (-iter / brojIteracija)) if iter / brojIteracija > 0.4 else P[1]
        c2 = P[2] + 1 + np.e ** (-iter / brojIteracija) if iter / brojIteracija > 0.4 else P[1]
        for i in range(0, brojCestica):
            r1, r2 = np.random.uniform(0, 1.0000000001, 2)
            v_novo = omega * v[i] + c1 * r1 * (pb[i] - x[i]) + c2 * r2 * (gb - x[i])
            x_novo = x[i] + v_novo
            veciOdOpsega = x_novo[x_novo > slika.max()]
            manjiOdOpsega = x_novo[x_novo < slika.min()]
            if len(veciOdOpsega) > 0 or len(manjiOdOpsega) > 0:
                continue
            f_x = f(x_novo, slika)
            if f_x < f(pb[i], slika):
                pb[i] = np.copy(x_novo)
            if f_x < f(gb, slika):
                gb = np.copy(x_novo)
            x[i] = x_novo
            v[i] = v_novo
        SSEErr.append(SSE(gb, slika))
        avgDist.append(prosjecnaUdaljenost(gb, slika))
        iter += 1
    return {"rez": gb, "SSE": SSEErr, "avgDist": avgDist}


brojIteracija = 100
brojCestica = 30
brojKlastera = 3
parametri = [0.9, 2, 2]

F1A1 = PSO(funkcijaDistance1, slika, brojCestica, brojIteracija, brojKlastera, parametri)
F2A1 = PSO(funkcijaDistance2, slika, brojCestica, brojIteracija, brojKlastera, parametri)
F3A1 = PSO(funkcijaDistance3, slika, brojCestica, brojIteracija, brojKlastera, parametri)

F1A2 = PSO2(funkcijaDistance1, slika, brojCestica, brojIteracija, brojKlastera, parametri)
F2A2 = PSO2(funkcijaDistance2, slika, brojCestica, brojIteracija, brojKlastera, parametri)
F3A2 = PSO2(funkcijaDistance3, slika, brojCestica, brojIteracija, brojKlastera, parametri)
plt.title(label="SSE")
plt.xlabel("Iterations")
plt.ylabel("SSE")
plt.plot(F1A1["SSE"], c="b", label="F1 Alg 1")
plt.plot(F2A1["SSE"], c="r", label="F2 Alg 1")
plt.plot(F3A1["SSE"], c="g", label="F3 Alg 1")
plt.plot(F1A2["SSE"], c="yellow", label="F1 Alg 2")
plt.plot(F2A2["SSE"], c="cyan", label="F2 Alg 2")
plt.plot(F3A2["SSE"], c="purple", label="F3 Alg 2")
plt.legend(loc="upper right")
plt.xlim([0, 130])
plt.show()

plt.title(label="AvgDist")
plt.xlabel("Iterations")
plt.ylabel("AvgDist")
plt.plot(F1A1["avgDist"], c="b", label="F1 Alg 1")
plt.plot(F2A1["avgDist"], c="r", label="F2 Alg 1")
plt.plot(F3A1["avgDist"], c="g", label="F3 Alg 1")
plt.plot(F1A2["avgDist"], c="yellow", label="F1 Alg 2")
plt.plot(F2A2["avgDist"], c="cyan", label="F2 Alg 2")
plt.plot(F3A2["avgDist"], c="purple", label="F3 Alg 2")
plt.legend(loc="upper right")
plt.xlim([0, 130])
plt.show()

slika2 = np.copy(slika)
slika3 = np.copy(slika)
slika4 = np.copy(slika)
slika5 = np.copy(slika)
slika6 = np.copy(slika)

KC1 = DajBojeZaKlastere(F1A1["rez"])
KC2 = DajBojeZaKlastere(F2A1["rez"])
KC3 = DajBojeZaKlastere(F3A1["rez"])
KC4 = DajBojeZaKlastere(F1A2["rez"])
KC5 = DajBojeZaKlastere(F2A2["rez"])
KC6 = DajBojeZaKlastere(F3A2["rez"])

for i in range(len(slika)):
    for j in range(len(slika[0])):
        for k in range(len(F1A1["rez"])):
            if slika[i][j] in KC1[k]:
                slika[i][j] = int(F1A1["rez"][k])
            if slika2[i][j] in KC2[k]:
                slika2[i][j] = int(F2A1["rez"][k])
            if slika3[i][j] in KC3[k]:
                slika3[i][j] = int(F3A1["rez"][k])
            if slika4[i][j] in KC4[k]:
                slika4[i][j] = int(F1A2["rez"][k])
            if slika5[i][j] in KC5[k]:
                slika5[i][j] = int(F2A2["rez"][k])
            if slika6[i][j] in KC6[k]:
                slika6[i][j] = int(F3A2["rez"][k])

fig, axs = plt.subplots(3, 2)
axs[0, 0].imshow(slika, cmap='gray')
axs[1, 0].imshow(slika2, cmap='gray')
axs[2, 0].imshow(slika3, cmap='gray')
axs[0, 1].imshow(slika4, cmap='gray')
axs[1, 1].imshow(slika5, cmap='gray')
axs[2, 1].imshow(slika6, cmap='gray')
axs[0, 0].set_title(label="F1 Alg1")
axs[1, 0].set_title(label="F2 Alg1")
axs[2, 0].set_title(label="F3 Alg1")
axs[0, 1].set_title(label="F1 Alg2")
axs[1, 1].set_title(label="F2 Alg2")
axs[2, 1].set_title(label="F3 Alg2")
plt.subplots_adjust(hspace=0.5)
plt.show()
