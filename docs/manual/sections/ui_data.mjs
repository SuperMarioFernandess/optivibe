// docs/manual/sections/ui_data.mjs
// AUTO-SLICED (S-DOC-2) from manual_build_doc.js lines 193..205; content verbatim.
// Owns its bookmarks/links. Edit this module only; build.mjs orders it via manifest.mjs.
import {
  R, c, P, para, H, xr, flink, ext, links, sep, code, cap, bullets, nums, img, tbl, fml,
  AlignmentType, Paragraph, TextRun, PageBreak,
  REPO, FIG, INK, MUT, ACC, ACCD, LINE, CW,
} from "../shared.mjs";

export function section() {
  const K = [];
  K.push(H(2,"3.3 Импорт данных приборов (replay)","s_ui_data"));
  K.push(para([R("Можно подать собственную измеренную вибрацию: загрузчики ("), flink("src/optivibe/io/loaders.py","io/loaders.py"), R(") читают CSV, WAV и форматы приборов TDMS (NI), UFF/UNV-58, MATLAB MAT, HDF5 и возвращают тот же контракт "), c("Excitation"), R(". Бэкенды ("), c("nptdms"), R("/"), c("pyuff"), R("/"), c("h5py"), R(") подключаются лениво за extra "), c("[io-formats]"), R("; их отсутствие даёт подсказку об установке.")]));
  K.push(...bullets([
    [R("Единицы приводятся к СИ на границе загрузчика: "), c("m/s^2"), R(" / "), c("g"), R(" / "), c("V"), R(" + чувствительность, либо "), c("auto"), R(" по метке файла; конфликт явной единицы с меткой — громкая ошибка.")],
    [R("Для 2-D записей выбирается канал (длинная ось — время); для MAT/HDF5 источник частоты дискретизации указывается явно.")],
    [R("Replay включается через "), c("excitation.kind: tdms|uff|mat|hdf5"), R(" (см. "), flink("examples/replay_tdms.yaml"), R(").")],
  ]));
  K.push(para([R("Форматы и их бэкенды (реестр загрузчиков "), c("LOADER_REGISTRY"), R(", "), flink("src/optivibe/io/loaders.py","io/loaders.py"), R("):")]));
  K.push(tbl(
    ["kind","Формат","Бэкенд","Частота дискретизации из","Метка единиц"],
    [
      ["csv","CSV-таблица","встроенный","столбец времени или fs_hz","поле units"],
      ["wav","WAV PCM","scipy (ядро)","заголовок файла","отображение в полную шкалу"],
      ["tdms","NI TDMS","nptdms (extra)","wf_increment или fs_hz","unit_string канала"],
      ["uff","UFF/UNV-58","pyuff (extra)","abscissa_inc или fs_hz","метка ординаты"],
      ["mat","MATLAB v4/v5/v7","scipy (ядро)","fs_hz или fs_key","нет (задать units)"],
      ["hdf5","HDF5 .h5/.hdf5","h5py (extra)","fs_hz или fs_attr","атрибут датасета"],
    ],
    [900,2100,1700,2860,1800]
  ));
  K.push(para([R("Extra "), c("[io-formats]"), R(" ставит "), c("nptdms"), R(" + "), c("pyuff"), R(" + "), c("h5py"), R(" (MAT и CSV/WAV — на ядре). MATLAB v7.3 — это HDF5 внутри: читать загрузчиком "), c("hdf5"), R(". Без extra загрузчик падает с подсказкой об установке, остальное ПО работает (см. "), R("docs/data-import.md",{size:18,color:MUT}), R(").")]));



  return K;
}
