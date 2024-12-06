
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter
from scipy.fft import fft
from skimage.morphology import skeletonize
from scipy.signal import welch

DIRECTIONS = [
  (0,   1),  # 0: Destra
  (-1,  1),  # 1: Alto a destra
  (-1,  0),  # 2: Alto
  (-1, -1),  # 3: Alto a sinistra
  (0,  -1),  # 4: Sinistra
  (1,  -1),  # 5: Basso a sinistra
  (1,   0),  # 6: Basso
  (1,   1),  # 7: Basso a destra
]

DIRECTIONAL_CLASSES = [
  (0, 0),  # Destra, Destra
  (0, 1),  # Destra, Alto a destra
  (0, 7),  # Destra, Basso a destra
  (7, 7),  # Basso a destra, Basso a destra
  (7, 0),  # Basso a destra, Destra
  (7, 6),  # Basso a destra, Basso
  (6, 6),  # Basso, Basso
  (6, 7),  # Basso, Basso a destra
  (6, 5),  # Basso, Basso a sinistra
  (5, 5),  # Basso a sinistra, Basso a sinistra
  (5, 6),  # Basso a sinistra, Basso
  (5, 4)   # Basso a sinistra, Sinistra
]

DIRECTIONAL_LABELS = [
  "→ →",  # (0, 0)
  "→ ↗",  # (0, 1)
  "→ ↘",  # (0, 7)
  "↘ ↘",  # (7, 7)
  "↘ →",  # (7, 0)
  "↘ ↓",  # (7, 6)
  "↓ ↓",  # (6, 6)
  "↓ ↘",  # (6, 7)
  "↓ ↙",  # (6, 5)
  "↙ ↙",  # (5, 5)
  "↙ ↓",  # (5, 6)
  "↙ ←"   # (5, 4)
]

# BINARIZZAZIONE

def load_grapheme_image(img):
  image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
  original_height, original_width = image.shape
  return image

def binarize_grapheme_image(image, threshold=127):
  _, bin_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
  return bin_image

def scale_grapheme_image(image, dim=0, scale_by_width=True):
  original_height, original_width = image.shape
  # Esegui la scalatura solo se dim > 0
  if dim > 0:
    if scale_by_width:
      new_width = dim
      scale_ratio = new_width / original_width
      new_height = int(original_height * scale_ratio)
      image = cv2.resize(image, (new_width, new_height))
  return image

def rotate_grapheme_image(img, angle=0):
  # Rotazione in senso antiorario in base all'angolo
  if angle != 0:
    # Calcola le dimensioni del canvas necessarie
    # per contenere l'immagine ruotata
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Calcola l'angolo di rotazione
    # per ottenere le dimensioni del canvas
    cos_angle = np.abs(rotation_matrix[0, 0])
    sin_angle = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin_angle) + (w * cos_angle))
    new_h = int((h * cos_angle) + (w * sin_angle))
    # Adatta la matrice di rotazione
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    # Applica la rotazione
    rotated_img = cv2.warpAffine(img, rotation_matrix, (new_w, new_h))
  else:
    rotated_img = img
  return rotated_img

def extract_grapheme_edge(image):
  sobel_x = cv2.Sobel(image, cv2.CV_64F, 1,0, ksize=3)
  sobel_y = cv2.Sobel(image, cv2.CV_64F, 0,1, ksize=3)
  edges = np.sqrt(sobel_x**2 + sobel_y**2)
  _, edge_binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
  edge_binary = (edge_binary > 0).astype(np.uint8)
  return edge_binary

def thin_grapheme_edge(image_edge):
  image_edge = image_edge > 0
  skeleton = skeletonize(image_edge)
  skeleton = np.uint8(skeleton) * 1
  return skeleton

# SEGMENTAZIONE

def direction(p1, p2):
  delta = (p2[0] - p1[0], p2[1] - p1[1])
  if delta in DIRECTIONS:
    return DIRECTIONS.index(delta)
  return None

def is_valid_point(p, edge_data):
  rows, cols = edge_data.shape
  return 0<=p[0]<rows and 0<=p[1]<cols and edge_data[p[0], p[1]]>0

def extend_segment(p, d, s, dir_sequence, visited, edge_data):
  while True:
      neighbors = [(p[0] + dx, p[1] + dy) for dx, dy in DIRECTIONS]
      valid_neighbors = [
          (p2, direction(p, p2))
          for p2 in neighbors
          if is_valid_point(p2, edge_data) and p2 not in visited
      ]
      if not valid_neighbors:
          break
      extended = False
      for p2, dir_p2 in valid_neighbors:
          if dir_p2 == d or dir_p2 == s:
              if dir_p2 == s and dir_sequence and dir_sequence[-1] == s:
                  continue
              p = p2
              visited.add(p)
              dir_sequence.append(dir_p2)
              extended = True
              break
      if not extended:
          break
  return dir_sequence, True

def valid_sequence_format(lst):
  in_sequence_of_ones = False
  for i in range(len(lst)):
      if lst[i] == 1:
          in_sequence_of_ones = True
      elif lst[i] == 0:
          if in_sequence_of_ones:
              in_sequence_of_ones = False
          else:
              return False
      else:
          return False
  return True

def detect_quasi_straight_segments(edge_data, n, s, l):
  Q = []
  visited = set()
  rows, cols = edge_data.shape
  for i in range(rows):
      for j in range(cols):
          if edge_data[i, j] != 0 and (i, j) not in visited:
              p1 = (i, j)
              for dir in DIRECTIONS:
                  p2 = (p1[0] + dir[0], p1[1] + dir[1])
                  if is_valid_point(p2,edge_data) and direction(p1, p2) == n:
                      dir_sequence = []
                      visited.add(p1)
                      visited.add(p2)
                      dir_sequence, valid = extend_segment(
                          p2, n, s, dir_sequence, visited, edge_data
                      )
                      if valid:
                          dir_sequence, valid = extend_segment(
                              p1,
                              (n + 4) % 8,
                              (s + 4) % 8,
                              dir_sequence[::-1],
                              visited,
                              edge_data,
                          )
                      if valid and len(dir_sequence) >= l:
                          Q.append(dir_sequence)
  return Q

def segment_grapheme_edge(img, seg_length=2):
  segments = []
  for dominant, deviation in DIRECTIONAL_CLASSES:
    detected_segment = detect_quasi_straight_segments(img, dominant,
                                                      deviation,
                                                      seg_length)
    segments.append(detected_segment)
  return segments

# ANALISI ENERGETICA

def binarize_grapheme_edge(grapheme):
  binarized_grapheme = []
  for class_index, directional_class in enumerate(grapheme):
      dominant, deviation = DIRECTIONAL_CLASSES[class_index]
      binarized_class = []
      for segment in directional_class:
          binarized_segment = [
              1 if code == dominant else 0 for code in segment]
          if valid_sequence_format(binarized_segment):
              binarized_class.append(binarized_segment)
      binarized_grapheme.append(binarized_class)
  return binarized_grapheme

def aggregate_edge_classes(binarized_grapheme_edge):
  aggregated_edge_classes = []
  for binarized_class in binarized_grapheme_edge:
      if len(binarized_class) > 0:
          aggregated_edge_classes.append((np.concatenate(binarized_class)).tolist())
      else:
          aggregated_edge_classes.append([])
  return aggregated_edge_classes

def calculate_gradients(aggregated_segment):
  return np.abs(np.diff(aggregated_segment))

def calculate_directionalities(segment_gradients, weight, k=0.693, angle=None):
  segment_gradients = np.array(segment_gradients)
  directionalities = np.exp(-k * segment_gradients) * np.exp(1j * segment_gradients * np.pi / 4) * np.exp(np.log(2))
  if angle is not None:
      directionalities *= np.exp(1j * angle)
  return directionalities

def calculate_directionality(directionalities):
  total_directionality = np.sum(directionalities)
  magnitude = np.abs(total_directionality)
  angle = np.angle(total_directionality)
  return total_directionality, magnitude, angle

def calculate_classes_directionalities(aggregated_segments, angle):
  cc_directionalities = []
  classes_directionalities = []
  i = 0
  for aggregated_segment in aggregated_segments:
      c_directionalities = []
      class_directionalities = []
      if len(aggregated_segment) > 0:
          gradients = calculate_gradients(aggregated_segment)
          directionalities = calculate_directionalities(gradients, angle)
          c_directionalities = directionalities.tolist()
          directionality, magnitude, angle = calculate_directionality(directionalities)
          class_directionalities.append((directionality, magnitude, angle))
      cc_directionalities.append(c_directionalities)
      classes_directionalities.append(class_directionalities)
      i = i + 1
  return classes_directionalities, cc_directionalities

def extract_directionalities(lists):
  result = []
  for sublist in lists:
    if sublist:
      result.append(sublist[0][0])
    else:
      result.append(0)
  return result

def calculate_total_directionalities(classes_directionalities):
  total_directionalities = []
  for class_directionalities in classes_directionalities:
      if len(class_directionalities):
          # Somma delle direzionalità complesse
          total_directionality, total_magnitude, total_angle = calculate_directionality([d[0] for d in class_directionalities])
          total_directionalities.append((total_directionality, total_magnitude, total_angle))
      else:
          total_directionalities.append((0, 0, 0))
  return total_directionalities

def calculate_class_magnitude(class_directionalities):
  # Considera solo la magnitudine
  return np.sqrt(np.sum(np.square(np.abs(class_directionalities))))

def calculate_class_energy(class_magnitude):
    return class_magnitude**2

def calculate_class_power(class_energy, class_length):
  return class_energy / class_length

def calculate_classes_magnitudes(classes_directionalities):
  classes_magnitudes = []
  for class_directionalities in classes_directionalities:
    if len(class_directionalities):
      # Considera solo le direzionalità
      class_magnitude = np.abs(calculate_class_magnitude([d[0] for d in class_directionalities]))
      classes_magnitudes.append(class_magnitude)
    else:
      classes_magnitudes.append(0)
  return classes_magnitudes

def calculate_classes_energies(classes_magnitudes):
  classes_energies = []
  for class_magnitude in classes_magnitudes:
    if class_magnitude:
      class_energy = calculate_class_energy(class_magnitude)
      classes_energies.append(class_energy)
    else:
      classes_energies.append(0)
  return classes_energies

def calculate_total_energy(classes_energies):
  total_energy = sum(classes_energies)
  return total_energy

def calculate_classes_powers(classes_energies, classes_magnitudes):
  classes_powers = []
  for i, class_energy in enumerate(classes_energies):
      if class_energy:
          class_power = calculate_class_power(class_energy, classes_magnitudes[i])
          classes_powers.append(class_power)
      else:
          classes_powers.append(0)
  return classes_powers

def calculate_total_power(classes_powers):
  total_power = sum(classes_powers)
  return total_power

def calculate_psd(segment_signal, fs=1.0):
  segment_signal = np.array(segment_signal)
  N = len(segment_signal)
  fft_result = np.fft.fft(segment_signal)
  psd = (2 / (N * fs)) * np.abs(fft_result[:N//2])**2
  freqs = np.fft.fftfreq(N, d=1/fs)
  positive_freqs = freqs[:N//2]
  return positive_freqs, psd

def calculate_psds(aggregated_segments):
  class_psds = []
  for aggregated_segment in aggregated_segments:
      if len(aggregated_segment) > 0:
        freqs, psd = calculate_psd(aggregated_segment)
      else:
        freqs = np.array([0])
        psd = np.array([0])
      class_psds.append((freqs, psd))
  return class_psds

def calculate_total_powers(psd_results):
  total_powers = []
  for freqs, psd in psd_results:
    # Somma tutti i valori della PSD per calcolare la potenza totale
    total_power = np.sum(psd)
    total_powers.append(total_power)
  return total_powers

def calculate_total_powers_exclude_dc(psd_results):
  total_powers = []
  for freqs, psd in psd_results:
      # Escludi la componente continua, che è la prima frequenza (0 Hz)
      # Somma tutti i valori di PSD escludendo il primo elemento (frequenza 0)
      total_power = np.sum(psd[1:])
      total_powers.append(total_power)
  return total_powers

def calculate_inverted_max_psds(class_psds):
  inverted_max_psds = []
  for idx, (freqs, psd) in enumerate(class_psds):
      max_psd = np.max(psd)
      inverted_max_psds.append(max_psd)
  return inverted_max_psds

def calculate_rapporto_powers_inverted_max_psds(classes_powers, inverted_max_psds):
  rapporto_powers_inverted_max_psds = []
  if len(classes_powers) == len(inverted_max_psds):
      for idx in range(len(inverted_max_psds)):
          if classes_powers[idx] != 0:
              rapporto = inverted_max_psds[idx]/classes_powers[idx]
              rapporto_powers_inverted_max_psds.append(rapporto)
          else:
              rapporto_powers_inverted_max_psds.append(0)
  return rapporto_powers_inverted_max_psds

def calculate_total_inverted_psd(inverted_max_psds):
  total_inverted_psd = sum(inverted_max_psds)
  return total_inverted_psd

def show_image(img):
  plt.imshow(img, cmap='gray')
  plt.axis('off')
  plt.show()

def plot_powers(powers, labels, colors):
  n = len(labels)
  assert len(powers) == n, "La lunghezza delle potenze deve corrispondere alla lunghezza delle etichette"
  assert len(colors) >= n, "Il numero di colori deve essere almeno uguale al numero di etichette"
  ind = np.arange(n)
  width = 0.4
  fig, ax = plt.subplots(figsize=(10, 6))
  bars = ax.bar(ind, powers, width, color=colors[:n], label='Potenza')
  ax.set_xlabel('Classe Direzionale')
  ax.set_ylabel('Potenza')
  ax.set_title('Potenza per Classe Direzionale')
  ax.set_xticks(ind)
  ax.set_xticklabels(labels)
  plt.tight_layout()
  plt.show()

def plot_psds(class_psds, colors):
  plt.figure(figsize=(10, 6))
  for idx, (freqs, psd) in enumerate(class_psds):
    if idx < len(DIRECTIONAL_LABELS):
      label = DIRECTIONAL_LABELS[idx]
    else:
      label = f"Class {idx + 1} (No Label)"
    plt.plot(freqs, psd, label=label, color=colors[idx % len(colors)])
  plt.xlabel('Frequenza')
  plt.ylabel('PSD')
  plt.title('PSD per Classe Direzionale')
  plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
  plt.grid(True)
  plt.show()

def plot_inverted_psds(class_psds, powers, colors):
  plt.figure(figsize=(10, 6))
  for idx, (freqs, psd) in enumerate(class_psds):
    max_psd = powers[idx] #np.max(psd)
    inverted_psd = max_psd - (psd)
    if idx < len(DIRECTIONAL_LABELS):
      label = DIRECTIONAL_LABELS[idx]
    else:
      label = f"Class {idx + 1} (No Label)"
    plt.plot(freqs, inverted_psd, label=label, color=colors[idx % len(colors)])
  plt.xlabel('Frequenza')
  plt.ylabel('PSD invertita')
  plt.title('PSD invertita per Classe Direzionale')
  plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
  plt.grid(True)
  plt.show()

def serialize_complex(obj):
  if isinstance(obj, complex):
      return [obj.real, obj.imag]
  elif isinstance(obj, (np.integer, np.floating)):
      return obj.item()
  raise TypeError(f"Tipo non serializzabile: {type(obj)}")

def main(path, dim, angle, min_segment_len, colors, test=None):
  grapheme = load_grapheme_image(path)
  grapheme = binarize_grapheme_image(grapheme)
  grapheme = scale_grapheme_image(grapheme, dim, True)
  grapheme = extract_grapheme_edge(grapheme)
  grapheme = thin_grapheme_edge(grapheme)
  grapheme = rotate_grapheme_image(grapheme, angle=angle)
  segmented_grapheme = segment_grapheme_edge(grapheme, min_segment_len)
  binarized_grapheme_edge = binarize_grapheme_edge(segmented_grapheme)
  aggregated_segments = aggregate_edge_classes(binarized_grapheme_edge)
  classes_directionalities, cc_directionalities = calculate_classes_directionalities(aggregated_segments, angle)
  #total_directionalities = calculate_total_directionalities(classes_directionalities)
  classes_magnitudes = calculate_classes_magnitudes(classes_directionalities)
  classes_energies = calculate_classes_energies(classes_magnitudes)
  total_energy = calculate_total_energy(classes_energies)
  classes_powers = calculate_classes_powers(classes_energies, classes_magnitudes)
  total_power = calculate_total_power(classes_powers)
  psds = calculate_psds(aggregated_segments)
  #psds = calculate_psds(cc_directionalities)
  total_psd_powers = calculate_total_powers(psds)
  total_powers_exclude_dc = calculate_total_powers_exclude_dc(psds)
  inverted_max_psds = calculate_inverted_max_psds(psds)
  rapporto_powers_inverted_max_psds = calculate_rapporto_powers_inverted_max_psds(classes_powers, inverted_max_psds)
  total_inverted_psd = calculate_total_inverted_psd(inverted_max_psds)
  rapporto_totale = total_power/total_inverted_psd
  report = [
      f'path: {path}',
      f'angle: {angle}',
      f'classes_directionalities: {classes_directionalities}',
      f'total_directionalities: {total_directionalities}',
      f'classes_magnitudes: {classes_magnitudes}',
      f'classes_energies: {classes_energies}',
      f'classes_powers: {classes_powers}',
      f'inverted_max_psds: {inverted_max_psds}',
      f'total_energy: {total_energy}',
      f'total_power: {total_power}',
      f'total_inverted_psd: {total_inverted_psd}',
      f'rapporto_powers_inverted_max_psds: {rapporto_powers_inverted_max_psds}',
      f'rapporto_totale: {rapporto_totale}'
  ]
  with open(f'{path}.txt', 'a') as file:
      json.dump(report, file, default=serialize_complex)
      file.write('\n')
  show_image(grapheme)
  plot_powers(classes_powers, DIRECTIONAL_LABELS, colors)
  plot_psds(psds, colors)
  plot_inverted_psds(psds, total_psd_powers, colors)

main(path="imgs/grapheme.png",
     dim=400,
     angle=0,
     min_segment_len=5,
     colors=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
             'orange', 'purple', 'pink', 'brown', 'gray'],
     test=None)
