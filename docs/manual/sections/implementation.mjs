// docs/manual/sections/implementation.mjs
// AUTO-SLICED (S-DOC-2) from manual_build_doc.js lines 399..643; content verbatim.
// Owns its bookmarks/links. Edit this module only; build.mjs orders it via manifest.mjs.
import {
  R, c, P, para, H, xr, flink, ext, links, sep, code, cap, bullets, nums, img, tbl, fml,
  AlignmentType, Paragraph, TextRun, PageBreak,
  REPO, FIG, INK, MUT, ACC, ACCD, LINE, CW,
} from "../shared.mjs";

export function section() {
  const K = [];
  K.push(H(1,"7. Реализация в коде","ch7"));
  K.push(para([R("Ниже — представительные блоки, отражающие задокументированные контракты, протоколы и API (имена, поля, ключи реестров). Авторитетный источник — репозиторий; здесь показаны ключевые места реализации с ссылками на файлы. Стиль соответствует конвенциям проекта (документ 10): строгая типизация, numpy-докстринги, единицы СИ.")]));

  K.push(H(2,"7.1 Контракты данных","s_impl_types"));
  K.push(para([flink("src/optivibe/core/types.py","core/types.py"), R(" — иммутабельные контракты с однократной валидацией формы/частоты:")]));
  K.push(...code([
    "from dataclasses import dataclass",
    "import numpy as np",
    "import numpy.typing as npt",
    "",
    "FloatArray = npt.NDArray[np.float64]",
    "",
    "@dataclass(frozen=True, slots=True)",
    "class Excitation:",
    '    """3D-ускорение основания. Поля в СИ (м/с²); fs в Гц."""',
    "    a_x: FloatArray",
    "    a_y: FloatArray",
    "    a_z: FloatArray",
    "    fs: float",
    "    seed: int | None = None",
    "    meta: dict[str, object] = field(default_factory=dict)",
    "",
    "    def __post_init__(self) -> None:",
    "        n = self.a_x.shape[0]",
    "        if not (self.a_y.shape[0] == self.a_z.shape[0] == n):",
    '            raise ValueError(\"оси a_x/a_y/a_z должны быть одной длины\")',
    "        if self.fs <= 0:",
    '            raise ValueError(\"fs должна быть положительной\")',
    "",
    "@dataclass(frozen=True, slots=True)",
    "class TipState:",
    '    """Вектор состояния торца (ICD, документ 04): смещения [м], наклоны [рад]."""',
    "    dx: FloatArray; dy: FloatArray; dz: FloatArray",
    "    theta_x: FloatArray; theta_y: FloatArray",
    "    fs: float",
  ]));
  K.push(links([ xr("Контракты — обзор → §5.3","s_arch_contracts"), sep(), flink("src/optivibe/core/types.py") ]));

  K.push(H(2,"7.2 Реестр сменных реализаций","s_impl_registry"));
  K.push(para([flink("src/optivibe/core/registry.py","core/registry.py"), R(" — типобезопасный реестр стадий по ключу из конфига:")]));
  K.push(...code([
    "from typing import Callable, TypeVar, Generic",
    "",
    "T = TypeVar(\"T\")",
    "",
    "class Registry(Generic[T]):",
    '    """Отображение строкового ключа в фабрику реализации стадии."""',
    "    def __init__(self, family: str) -> None:",
    "        self._family = family",
    "        self._items: dict[str, Callable[..., T]] = {}",
    "",
    "    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:",
    "        def deco(factory: Callable[..., T]) -> Callable[..., T]:",
    "            if key in self._items:",
    "                raise KeyError(f\"{self._family}: ключ {key!r} уже зарегистрирован\")",
    "            self._items[key] = factory",
    "            return factory",
    "        return deco",
    "",
    "    def create(self, key: str, *args: object, **kw: object) -> T:",
    "        if key not in self._items:",
    "            raise KeyError(f\"{self._family}: неизвестный ключ {key!r}; \"",
    "                           f\"доступны {sorted(self._items)}\")",
    "        return self._items[key](*args, **kw)",
    "",
    "OPTICS_REGISTRY: Registry[OpticsStage] = Registry(\"optics\")",
    "DSP_REGISTRY: Registry[DspStage] = Registry(\"dsp\")",
  ]));
  K.push(links([ xr("Реестры — ключи → §5.4","s_arch_registry"), sep(), flink("src/optivibe/core/registry.py") ]));

  K.push(H(2,"7.3 Протокол стадии и регистрация","s_impl_stage"));
  K.push(para([flink("src/optivibe/core/stages.py","core/stages.py"), R(" определяет протоколы; реализация регистрируется декоратором:")]));
  K.push(...code([
    "from typing import Protocol",
    "",
    "class MechanicsStage(Protocol):",
    "    def run(self, excitation: Excitation, variant: VariantConfig) -> TipState: ...",
    "",
    "@MECHANICS_REGISTRY.register(\"modal\")",
    "def _make_modal() -> MechanicsStage:",
    "    return ModalMechanics()",
  ]));

  K.push(H(2,"7.4 Генератор возбуждения (пример: sine)","s_impl_exc"));
  K.push(para([flink("src/optivibe/excitation/generators.py","excitation/generators.py"), R(" — генераторы за протоколом "), c("ExcitationSource.generate"), R(":")]));
  K.push(...code([
    "from optivibe.core.units import G0  # 9.80665 м/с²",
    "",
    "@EXCITATION_REGISTRY.register(\"sine\")",
    "class SineSource:",
    "    def generate(self, spec: SineSpec, *, seed: int | None) -> Excitation:",
    "        n = round(spec.duration_s * spec.fs_hz)",
    "        t = np.arange(n) / spec.fs_hz",
    "        a = spec.amplitude_g * G0 * np.sin(2 * np.pi * spec.frequency_hz * t)",
    "        zeros = np.zeros(n)",
    "        axes = {\"x\": (a, zeros, zeros), \"y\": (zeros, a, zeros),",
    "                \"z\": (zeros, zeros, a)}",
    "        ax, ay, az = axes[spec.axis]",
    "        return Excitation(ax, ay, az, fs=spec.fs_hz, seed=seed)",
  ]));
  K.push(links([ xr("Алгоритм возбуждения → §6.1","s_algo_exc"), sep(), flink("src/optivibe/excitation/") ]));

  K.push(H(2,"7.5 Механика: f₁, H_lat(f), частотный решатель","s_impl_mech"));
  K.push(para([flink("src/optivibe/mechanics/cantilever.py","mechanics/cantilever.py"), R(" — производные величины и частотный решатель (документ 02/05):")]));
  K.push(...code([
    "def first_natural_frequency_hz(geom: BeamGeometry, mat: Material) -> float:",
    '    """f1 = (β1 L)² / (2π) · √(EI / ρSL⁴). См. документ 02 §3.2."""',
    "    beta1_l = 1.8751",
    "    return beta1_l**2 / (2*np.pi) * np.sqrt(",
    "        mat.E * geom.I / (mat.rho * geom.S * geom.L**4))",
    "",
    "def transfer_D(freq_hz: FloatArray, f1: float, q: float) -> np.ndarray:",
    '    """Резонансный множитель D(f) = 1/[1 − (f/f1)² + i(f/f1)/Q]."""',
    "    r = freq_hz / f1",
    "    return 1.0 / (1.0 - r**2 + 1j * r / q)",
    "",
    "def first_mode_shape(xi: FloatArray, beta1_l: float = 1.8751) -> FloatArray:",
    '    """φ1(z/L), нормировка на единичный торец (для анимации изгиба, GUI)."""',
    "    s = ((np.cosh(beta1_l)+np.cos(beta1_l)) /",
    "         (np.sinh(beta1_l)+np.sin(beta1_l)))",
    "    phi = (np.cosh(beta1_l*xi)-np.cos(beta1_l*xi)",
    "           - s*(np.sinh(beta1_l*xi)-np.sin(beta1_l*xi)))",
    "    return phi / phi[-1]",
  ]));
  K.push(...code([
    "class ModalMechanics:  # ключ 'modal'",
    "    def run(self, exc: Excitation, variant: VariantConfig) -> TipState:",
    "        f1 = first_natural_frequency_hz(variant.geom, variant.material)",
    "        h_qs = quasistatic_gain(variant)          # H_lat^QS, м/(м/с²)",
    "        freq = np.fft.rfftfreq(exc.a_x.size, 1/exc.fs)",
    "        H = h_qs * transfer_D(freq, f1, variant.q_total)",
    "        dx = np.fft.irfft(np.fft.rfft(exc.a_x) * H, n=exc.a_x.size)",
    "        theta_y = (1.377 / variant.geom.L) * dx   # связь наклон–смещение",
    "        # ... dy (та же АЧХ), dz (квазистатика по оси z) ...",
    "        return TipState(dx, dy, dz, theta_x, theta_y, fs=exc.fs)",
  ]));
  K.push(links([ xr("Физика → §4.2","s_phys_mech"), sep(), xr("Алгоритм → §6.2","s_algo_mech"), sep(), flink("src/optivibe/mechanics/cantilever.py") ]));


  K.push(H(2,"7.6 Оптика: компоненты η","s_impl_opt"));
  K.push(para([flink("src/optivibe/optics/cylinder.py","optics/cylinder.py"), R(" — замкнутые формы η = η_x·η_y (документ 03 §4):")]));
  K.push(...code([
    "class CylinderOptics:  # ключ 'cylinder'",
    "    def run(self, tip: TipState, variant: VariantConfig) -> OpticalResponse:",
    "        m = CylinderModel.from_config(variant)   # валидаторы R_c≥5w0, w(A)≤R_c/3",
    "        eta_x, eta_y = m.eta_components(tip)",
    "        eta = eta_x * eta_y",
    "        return OpticalResponse(eta, eta_x, eta_y, bias=m.eta0, fs=tip.fs)",
    "",
    "    # внутри CylinderModel.eta_components:",
    "    #   g      = A + dz",
    "    #   dx_eff = dx0 + dx + (R_c + g) * theta_y      # 04 §3",
    "    #   d_x    = (2*g / R_c) * dx_eff;  alpha_x = (2 / R_c) * dx_eff",
    "    #   eta_x  = eta_par_x(g) * exp(-(d_x/w0)**2 - (alpha_x/theta0)**2)",
    "    #   eta_y  = 1/sqrt(1 + (g/zR)**2)               # от dy НЕ зависит (симметрия)",
  ]));
  K.push(links([ xr("Физика → §4.3","s_phys_opt"), sep(), xr("Алгоритм → §6.3","s_algo_opt"), sep(), flink("src/optivibe/optics/cylinder.py") ]));

  K.push(H(2,"7.7 Детектор: фототок, шумы, АЦП","s_impl_det"));
  K.push(para([flink("src/optivibe/detector/photodiode.py","detector/photodiode.py"), R(" — фототок + шумовой бюджет (документ 07):")]));
  K.push(...code([
    "class PhotodiodeDetector:  # ключ 'photodiode'",
    "    def __init__(self, options: DetectorOptions, *, seed: int | None) -> None:",
    "        self._opt = options",
    "        sub = np.random.SeedSequence([seed, 0x44455430])  # субсид детектора",
    "        self._rng = np.random.default_rng(sub)",
    "",
    "    def run(self, optical: OpticalResponse, variant: VariantConfig) -> DetectorOutput:",
    "        cfg = variant.detector",
    "        i = cfg.responsivity * cfg.power_w * (cfg.R1 + cfg.rho * optical.eta)",
    "        i_dc = cfg.responsivity * cfg.power_w * (cfg.R1 + cfg.rho * optical.bias)",
    "        b = optical.fs / 2.0                       # полоса Найквиста",
    "        shot = np.sqrt(2*E_CHARGE*i_dc*b)          # СКЗ дробового тока",
    "        rin  = i_dc*np.sqrt(cfg.rin_level*b)",
    "        if cfg.balanced:                            # подавление RIN опорным каналом",
    "            rin *= 10**(-cfg.cmrr_db/20)",
    "        noise = self._rng.normal(0, np.hypot(shot, rin), i.size)",
    "        samples = adc_quantize(i - i_dc + noise, cfg) + i_dc   # AC-связь + квант.",
    "        return DetectorOutput(samples, optical.fs, i_dc, units=\"A\", noise=...)",
  ]));
  K.push(links([ xr("Физика шумов → §4.4","s_phys_det"), sep(), xr("Алгоритм → §6.4","s_algo_det"), sep(), flink("src/optivibe/detector/photodiode.py") ]));

  K.push(H(2,"7.8 DSP: калибровка, кинематика, чувствительность","s_impl_dsp"));
  K.push(para([flink("src/optivibe/dsp/calibration.py","dsp/calibration.py"), R(" и "), flink("src/optivibe/dsp/kinematics.py","dsp/kinematics.py"), R(" — восстановление a→v→x:")]));
  K.push(...code([
    "def calibrate_acceleration(det: DetectorOutput, variant, opts,",
    "                           model: SensitivityModel | None = None) -> FloatArray:",
    '    """I_AC / s_target → ускорение целевой оси. model=None ⇒ путь v1."""',
    "    i_ac = det.samples - det.dc_level            # AC-связь уже сняла пьедестал",
    "    s = (model.at(nominal_tip(variant)).value if model is not None",
    "         else s_target_scalar(variant))          # А/(м/с²), знак учтён",
    "    return i_ac / s",
    "",
    "def integrate(a: FloatArray, fs: float, opts: DspOptions) -> tuple[FloatArray, FloatArray]:",
    '    """a→v→x. Реестр INTEGRATOR_REGISTRY: frequency (1/jω + ВЧ-маска) | time."""',
    "    integ = INTEGRATOR_REGISTRY.create(opts.integrator)",
    "    v = integ.once(a, fs, opts)",
    "    x = integ.once(v, fs, opts)",
    "    return v, x",
  ]));
  K.push(para([flink("src/optivibe/dsp/sensitivity.py","dsp/sensitivity.py"), R(" — переключаемая чувствительность (вектор-готовая сигнатура "), c(".at()"), R("):")]));
  K.push(...code([
    "@SENSITIVITY_REGISTRY.register(\"static\")          # умолчание = v1",
    "class StaticSensitivity:",
    "    def at(self, state: TipPoint, freq_hz: FloatArray | None = None) -> Sensitivity:",
    "        return Sensitivity(value=self._s_qs, target_axis=\"x\", freq_hz=freq_hz)",
  ]));
  K.push(links([ xr("Физика обратной задачи → §4.5","s_phys_e2e"), sep(), xr("Алгоритм → §6.5","s_algo_dsp"), sep(), xr("Чувствительность → §6.6","s_algo_sens"), sep(), flink("src/optivibe/dsp/") ]));

  K.push(H(2,"7.9 Бюджет NEA","s_impl_nea"));
  K.push(para([flink("src/optivibe/dsp/nea.py","dsp/nea.py"), R(" — приведение шума ко входу (документ 07):")]));
  K.push(...code([
    "def nea_spectrum(det: DetectorOutput, s_target: float) -> NeaResult:",
    '    """NEA(f) = √(PSD_тока) / |s_target|  [ (м/с²)/√Гц ]; + разложение по вкладам."""',
    "    n = det.noise",
    "    total = np.sqrt(n.psd_total_a2_hz) / abs(s_target)",
    "    shot  = np.sqrt(n.psd_shot_a2_hz)  / abs(s_target)",
    "    rin   = np.sqrt(n.psd_rin_a2_hz)   / abs(s_target)",
    "    return NeaResult(total=total, shot=shot, rin=rin, johnson=...)",
  ]));

  K.push(H(2,"7.10 GUI: расчёт вне UI-потока","s_impl_gui"));
  K.push(para([R("Qt-free задача ("), flink("src/optivibe/gui/workers/jobs.py","gui/workers/jobs.py"), R(") вызывает ядро; "), c("JobWorker"), R(" исполняет её на "), c("QThread"), R(", "), c("JobController"), R(" управляет жизненным циклом и сигналами:")]));
  K.push(...code([
    "@dataclass(frozen=True)",
    "class ScenarioJob:                       # Qt-free: можно тестировать без Qt",
    "    config: ScenarioConfig",
    "    def __call__(self, progress) -> VibrationResult:",
    "        return run_scenario(self.config)  # тяжёлый расчёт — в рабочем потоке",
    "",
    "class JobController(QObject):",
    "    def submit(self, job) -> None:",
    "        self._thread = QThread()",
    "        self._worker = JobWorker(job)",
    "        self._worker.moveToThread(self._thread)",
    "        self._worker.finished.connect(self._on_finished)  # сигнал → UI-поток",
    "        self._worker.failed.connect(self._on_failed)",
    "        self._thread.started.connect(self._worker.run)",
    "        self._thread.start()              # окно остаётся отзывчивым",
  ]));
  K.push(links([ xr("GUI и потоки → §3.1","s_ui_threads"), sep(), flink("src/optivibe/gui/controllers/job_controller.py"), sep(), flink("src/optivibe/gui/workers/job_worker.py") ]));


  return K;
}
