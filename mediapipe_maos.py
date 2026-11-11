"""Etapa 1: rastreamento de maos com MediaPipe, sem reconhecimento de Libras.

Abre a webcam, detecta ate duas maos e mostra os 21 pontos (landmarks),
as conexoes e os vetores (x, y, z) normalizados de cada ponto.

Comandos:
    python mediapipe_maos.py
    python mediapipe_maos.py --camera 1 --maos 2

Na janela: Q/ESC sai | N liga/desliga indices | V liga/desliga
coordenadas dos pontos selecionados | P imprime todos os vetores no terminal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import time
import urllib.request

# Executar dentro da pasta do repositorio ou de qualquer outro diretorio.
BASE = Path(__file__).resolve().parent
MODELO = BASE / "modelos" / "hand_landmarker.task"
URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Conexoes entre as 21 articulacoes definidas pelo MediaPipe.
CONEXOES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
)


def garantir_modelo() -> Path:
    """Na primeira execucao baixa o modelo oficial de deteccao de maos."""
    if MODELO.is_file() and MODELO.stat().st_size > 100_000:
        return MODELO

    MODELO.parent.mkdir(parents=True, exist_ok=True)
    temporario = MODELO.with_suffix(".download")
    print("Baixando o modelo de maos do MediaPipe (somente na primeira vez)...")
    try:
        urllib.request.urlretrieve(URL_MODELO, temporario)
        if temporario.stat().st_size <= 100_000:
            raise ValueError("O arquivo baixado parece incompleto.")
        temporario.replace(MODELO)
    except Exception as erro:
        temporario.unlink(missing_ok=True)
        raise RuntimeError(
            "Nao foi possivel baixar o modelo. Baixe manualmente em:\n"
            f"{URL_MODELO}\n"
            f"e salve como {MODELO}\nDetalhes: {erro}"
        ) from erro
    return MODELO


def abrir_camera(cv2, indice: int):
    """Abre webcam local, usando DirectShow no Windows quando possivel."""
    if os.name == "nt":
        camera = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(indice)
    else:
        camera = cv2.VideoCapture(indice)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Nao foi possivel abrir a camera {indice}. "
            "Verifique permissoes e se outro programa usa a webcam."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    return camera


def desenhar_mao(cv2, quadro, pontos, mostrar_indices: bool) -> None:
    """Desenha um esqueleto a partir dos pontos normalizados x,y,z."""
    altura, largura = quadro.shape[:2]
    pixels = [(int(p.x * largura), int(p.y * altura)) for p in pontos]

    for inicio, fim in CONEXOES:
        cv2.line(quadro, pixels[inicio], pixels[fim], (60, 210, 60), 2, cv2.LINE_AA)
    for indice, (x, y) in enumerate(pixels):
        cv2.circle(quadro, (x, y), 5, (20, 80, 255), -1, cv2.LINE_AA)
        if mostrar_indices:
            cv2.putText(
                quadro, str(indice), (x + 7, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 2,
                cv2.LINE_AA,
            )


def sobrepor_texto(cv2, quadro, texto: str, y: int) -> None:
    cv2.putText(quadro, texto, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(quadro, texto, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 2, cv2.LINE_AA)


def imprimir_vetores(resultado) -> None:
    """Imprime os 21 vetores (x, y, z) de cada mao detectada neste quadro."""
    if not resultado.hand_landmarks:
        print("Nenhuma mao detectada neste quadro.")
        return
    for numero, pontos in enumerate(resultado.hand_landmarks, start=1):
        print(f"\nMAO {numero} - 21 pontos | x,y: normalizados | z: profundidade relativa")
        for indice, p in enumerate(pontos):
            print(f"  Ponto {indice:02d}: x={p.x:+.5f}, y={p.y:+.5f}, z={p.z:+.5f}")
        vetor = [valor for p in pontos for valor in (p.x, p.y, p.z)]
        print(f"Vetor achatado com {len(vetor)} valores: {vetor}")


def executar(indice_camera: int, numero_maos: int) -> None:
    # Importacoes aqui para que `python mediapipe_maos.py --help` funcione
    # mesmo antes de instalar as bibliotecas.
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except ImportError as erro:
        raise RuntimeError("Dependencias ausentes. Execute: pip install -r requirements.txt") from erro

    caminho_modelo = garantir_modelo()
    opcoes = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=caminho_modelo.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=numero_maos,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    camera = abrir_camera(cv2, indice_camera)
    indices = True
    mostrar_vetores = False
    ultimo_timestamp = -1
    print("Controles: Q ou ESC: sair | N: numeros | V: vetores | P: imprimir pontos")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(opcoes) as detector:
            while True:
                ok, quadro = camera.read()
                if not ok:
                    print("Nao foi possivel ler um quadro da webcam.")
                    break

                # Efeito espelho para facilitar o uso frente a webcam.
                quadro = cv2.flip(quadro, 1)
                rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
                entrada = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )
                # A API VIDEO exige timestamps estritamente crescentes.
                timestamp = max(int(time.monotonic() * 1000), ultimo_timestamp + 1)
                ultimo_timestamp = timestamp
                resultado = detector.detect_for_video(entrada, timestamp)

                maos = resultado.hand_landmarks
                for pontos in maos:
                    desenhar_mao(cv2, quadro, pontos, indices)

                sobrepor_texto(cv2, quadro, f"Maos detectadas: {len(maos)}", 28)
                sobrepor_texto(cv2, quadro, "Q: sair | N: indices | V: vetores | P: imprimir", 55)
                if mostrar_vetores and maos:
                    # Exibir alguns landmarks sem poluir a tela; P mostra TODOS.
                    pontos = maos[0]
                    for linha, i in enumerate((0, 4, 8, 12, 16, 20), start=0):
                        p = pontos[i]
                        sobrepor_texto(cv2, quadro,
                                       f"P{i:02d}: ({p.x:+.3f}, {p.y:+.3f}, {p.z:+.3f})",
                                       85 + 24 * linha)

                cv2.imshow("MediaPipe - 21 pontos da mao (sem Libras)", quadro)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), ord("Q"), 27):
                    break
                if tecla in (ord("n"), ord("N")):
                    indices = not indices
                if tecla in (ord("v"), ord("V")):
                    mostrar_vetores = not mostrar_vetores
                if tecla in (ord("p"), ord("P")):
                    imprimir_vetores(resultado)
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detecta e desenha 21 pontos por mao na webcam, sem classificar Libras."
    )
    parser.add_argument("--camera", type=int, default=0, help="Indice da webcam (padrao: 0)")
    parser.add_argument("--maos", type=int, choices=(1, 2), default=1,
                        help="Quantidade maxima de maos (padrao: 1)")
    args = parser.parse_args()
    executar(args.camera, args.maos)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as erro:
        raise SystemExit(f"ERRO: {erro}") from erro
