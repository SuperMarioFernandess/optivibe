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
  K.push(para([R("Генераторы живут по файлу на семейство ("), flink("src/optivibe/excitation/tonal.py","excitation/tonal.py"), R(", "), c("sweep.py"), R(", "), c("random_noise.py"), R(", "), c("shock.py"), R(", "), c("from_file.py"), R(", "), c("composite.py"), R("), все — за одним протоколом "), c("ExcitationSource.generate"), R(":")]));
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
  K.push(para([R("Модуляция несущей — опциональное поле "), c("SineSpec.modulation"), R(" (§7.13); при её отсутствии выражение выше исполняется дословно, поэтому прежние сценарии дают побайтово тот же массив.")]));
  K.push(links([ xr("Алгоритм возбуждения → §6.1","s_algo_exc"), sep(), xr("Композит и модуляция → §7.13","s_impl_composite"), sep(), flink("src/optivibe/excitation/") ]));

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


  K.push(H(3,"7.5-бис Демпфирование: вычислимая Q(L)","s_impl_damping"));
  K.push(para([flink("src/optivibe/mechanics/damping.py","mechanics/damping.py"), R(" — модель добротности (документ 02 §5, 07 §2.3; решения R-47/R-48). Фактические имена функций:")]));
  K.push(...code([
    "reynolds_number(constants, omega_rad_s) -> float        # Re = ρ_f·ω·R²/μ_f",
    "knudsen_number(constants) -> float                      # Kn ≈ 1.09e-3 (континуум)",
    "hydrodynamic_function(reynolds) -> complex              # Γ_r + i·Γ_i (Sader 1998)",
    "q_air(constants, length_m) -> float                     # (ρ + ρ_f·Γ_r)/(ρ_f·Γ_i)",
    "q_anchor(constants, length_m) -> float                  # 2.17·(L/D)³",
    "damping_budget(constants, length_m, *, vacuum=False)    # {air, anchor, structural, ted, total}",
    "q_total_model(constants, length_m, *, vacuum=False)     # 1/Q = Σ 1/Q_i",
  ]));
  K.push(para([R("Точка вызова — не конвейер, а резолвинг композиции ("), c("SystemConfig.resolve"), R("): если "), c("q_total"), R(" опущен, он вычисляется и квантуется до 6 значащих цифр (иначе последний бит был бы платформозависим). При Re < 5 функция предупреждает в лог (не падает): усечённая асимптотика теряет ~1 % за пределами рабочего окна L = 1–5 мм.")]));

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

  K.push(H(3,"7.6-бис Источник: ширина линии, видность, RIN","s_impl_source"));
  K.push(para([flink("src/optivibe/optics/source.py","optics/source.py"), R(" — спектр источника (документ 03 §f′, 07 §1.2; решения R-46, R-55…R-57):")]));
  K.push(...code([
    "linewidth_nu_hz(wavelength_m, linewidth_fwhm_m)      # Δν = c·Δλ/λ²",
    "coherence_length_m(wavelength_m, linewidth_fwhm_m)   # L_c = (2ln2/π)·λ²/Δλ",
    "fringe_visibility(gap_m, coherence_length_m)             # гаусс:  V = 2^(−(2A/L_c)²)",
    "fringe_visibility_lorentzian(gap_m, coherence_length_m)  # лоренц: V = 2^(−4A/L_c)",
    "min_gap_for_washout_m(L_c)             # 1.1246·L_c   (V < 0.03)",
    "min_gap_for_washout_lorentzian_m(L_c)  # 1.2647·L_c   (+12.5 %)",
    "rin_ase(delta_nu_hz, *, lineshape=None) / rin_ase_db_hz(...)   # κ/Δν",
    "RIN_KAPPA_BY_LINESHAPE = {rectangular: 2, gaussian: 1.3286, lorentzian: 0.6366}",
    "# режим measured (табличный спектр OSA):",
    "fringe_visibility_measured(...)      # прямая квадратура Винера–Хинчина",
    "coherence_time_measured_s(...)       # τ_c (по Парсевалю)",
    "effective_linewidth_measured_hz(...) # Δν_eff = 1/τ_c",
    "rin_ase_measured(...) / rin_ase_measured_db_hz(...)   # RIN = 2/Δν_eff",
  ]));
  K.push(para([R("Проводка — в "), flink("src/optivibe/core/config/subsystems.py","core/config/subsystems.py"), R(": валидаторы "), c("_check_noise_inputs"), R(" / "), c("_check_lineshape_inputs"), R(" (правила пар «форма ↔ вход»; DFB обязан задать RIN явно), "), c("_effective_rin_db_hz"), R(" (вывод RIN), "), c("_check_source_coherence"), R(" (ворота смыва на маршруте 2 — громкий отказ при V ≥ 0.03).")]));

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
  K.push(para([R("(Фрагмент выше — иллюстративный.) Фактический контракт после M-12: "), c("NeaResult"), R(" несёт поля "), c("nea_optical"), R(", "), c("nea_thermal"), R(" и "), c("nea_plateau = hypot(optical, thermal)"), R("; функции "), c("nea_from_detector(..., include_thermal=True)"), R(" и "), c("nea_spectrum(..., include_thermal=True)"), R(" подмешивают тепловую ветвь "), R("в домене ускорения", { bold: true }), R(" (она не токовая PSD). Источник значения — "), flink("src/optivibe/mechanics/thermal.py","mechanics/thermal.py"), R(":")]));
  K.push(...code([
    "kinetic_effective_mass(constants, L)      # m_eff = 0.2427·ρSL  — задаёт ЧАСТОТУ",
    "acceleration_effective_mass(constants, L) # M_a  = 0.5795·ρSL  — задаёт ПОЛ",
    "thermal_force_psd(constants, L, q_total)  # S_F = 4kB·T·m_eff·ω1/Q   [Н²/Гц]",
    "nea_thermal(constants, L, q_total)        # NEA_th = √(4kB·T·ω1/(Q·M_a))",
  ]));
  K.push(para([R("Разложение по ветвям в ускорительном домене — "), flink("src/optivibe/analysis/nea_budget.py","analysis/nea_budget.py"), R(" ("), c("{shot, rin, johnson, thermal, total}"), R("). Два намеренных «не» (решение R-54): "), c("nea_from_psd"), R(" (тракт измеренных записей) тепловую ветвь "), R("не добавляет", { bold: true }), R(" — иначе двойной учёт; синтетический временной ряд детектора теплового движения "), R("не содержит", { bold: true }), R(". Флаг "), c("include_thermal=False"), R(" бит-в-бит воспроизводит поведение до M-12.")]));

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

  K.push(H(2,"7.11 Потоковый слой: причинный интегратор и бегущий спектр","s_impl_stream"));
  K.push(para([flink("src/optivibe/dsp/streaming.py","dsp/streaming.py"), R(" — реализация §6.5-бис. Три класса плюс драйвер реплея; батчевый путь не тронут, модуль строго аддитивен:")]));
  K.push(...code([
    "class LeakyIntegrator:                     # H(z) = (dt/2)(1+z⁻¹)/(1−α z⁻¹), α = e^{−2π f_c/fs}",
    "    def __init__(self, fs: float, f_c: float) -> None: ...",
    "    def process(self, block: FloatArray) -> FloatArray:  # состояние zi живёт между кадрами",
    "    def reset(self) -> None: ...",
    "",
    "class StreamingSpectrum:                   # кольцевой буфер + S[m] = β S[m−1] + (1−β)|X|²",
    "    def process(self, block: FloatArray) -> None: ...",
    "    def spectrum(self) -> Spectrum | None:  # None, пока не набран первый сегмент",
    "    @property",
    "    def ready(self) -> bool: ...",
    "",
    "class StreamingDsp:",
    "    def __init__(self, template: DetectorOutput, variant: VariantConfig,",
    "                 options: DspOptions, *, constants=None, sensitivity_model=None,",
    "                 nperseg: int = 1024, noverlap=None, avg_segments: int = 8,",
    "                 keep_history: bool = True, history_samples: int | None = None) -> None: ...",
    "    def process(self, sample_block: FloatArray) -> None: ...",
    "    def snapshot(self) -> VibrationResult:   # кадр: следы a/v/x, спектры, метрики, NEA",
    "    def note_dropped(self, count: int) -> None:",
    "    @property",
    "    def warmed(self) -> bool: ...            # осели ли причинные фильтры",
    "    @property",
    "    def dropped_samples(self) -> int: ...",
    "",
    "def replay_record(detector: DetectorOutput, variant: VariantConfig, options: DspOptions,",
    "                  *, block_size: int, ...) -> VibrationResult:   # драйвер приёмки",
  ]));
  K.push(para([R("Два параметра стоит понимать вместе. "), c("keep_history=True"), R(" держит полную запись (реплей и приёмка), "), c("keep_history=False"), R(" — ограниченный «осциллографный» буфер; "), c("history_samples"), R(" (keyword-only, только для ограниченного режима) задаёт окно "), R("отображаемого следа", { bold: true }), R(". Значение "), c("None"), R(" воспроизводит прежнее окно бит-в-бит — это закреплено тестом, а не декларацией; двигается "), R("только след", { bold: true }), R(", спектры, бегущие метрики и состояния интеграторов остаются побайтово теми же, а память ограничена предвыделенным кольцевым буфером, а не растущим списком.")]));
  K.push(para([R("Приёмочный якорь: "), c("replay_record"), R(" — это тот же цикл "), c("process()"), R("/"), c("snapshot()"), R(", поэтому golden эквивалентности поток↔батч покрывает и математику живого пути; отдельным тестом закреплено, что финальный кадр живого цикла побайтово равен "), c("replay_record"), R(" на той же записи.")]));
  K.push(links([ xr("Алгоритм → §6.5-бис","s_algo_stream"), sep(), xr("Живой режим → §3.6","s_live"), sep(), flink("src/optivibe/dsp/streaming.py") ]));

  K.push(H(2,"7.12 Ожидаемые пики: одна точка входа, реестр предикторов","s_impl_peaks"));
  K.push(para([flink("src/optivibe/analysis/expected_peaks.py","analysis/expected_peaks.py"), R(" — реализация §6.8. Место модуля выбрано по прецеденту бюджета NEA: это композиция нескольких доменов, а не стадия конвейера.")]));
  K.push(...code([
    "def predict_expected_peaks(scenario: ScenarioConfig, variant: VariantConfig,",
    "                           constants: Constants | None = None, *,",
    "                           kinds: Sequence[PeakKind] | None = None,",
    "                           max_harmonic: int = 3,",
    "                           sigma_factor: float = DEFAULT_SIGMA_FACTOR) -> ExpectedPeaks:",
    '    """Только конфигурация: временной ряд сюда физически не может попасть."""',
    "",
    "PEAK_KINDS = ('mode','harmonic','intermod','sideband','mains','alias','f_mount')",
    "PEAK_PREDICTOR_REGISTRY: Registry[tuple[ExpectedPeak, ...]]  # ветвь = регистрация",
    "",
    "@dataclass(frozen=True)",
    "class ExpectedPeak:",
    "    freq_hz: float; kind: PeakKind; label: str; explanation: str",
    "    amplitude_m_s2: float | None      # None там, где высота НЕ свойство конфигурации",
    "    threshold_m_s2: float | None; width_hz: float | None",
    "    order: int; source_freq_hz: float | None",
    "    @property",
    "    def significant(self) -> bool | None:   # None = сравнение не определено",
    "",
    "def amplitude_noise_threshold(...) -> float:   # k·√(2·Δf)·NEA(f)",
  ]));
  K.push(para([R("Контейнер "), c("ExpectedPeaks"), R(" несёт провенанс самого предсказания — f₁, эффективную Q, разрешение Δf, Найквиста, полку NEA, запрошенные ветви и полосу f₁/Q, — чтобы по списку линий можно было восстановить, из чего он получен. Инвариант «расчёт вне UI-потока» здесь держится "), R("структурно, а не по договорённости", { bold: true }), R(": функция принимает только сценарий и вариант и не способна получить временной ряд; добавление параметра «в форме записи» тихо аннулировало бы гарантию, о чём сказано прямо в докстрингах обеих точек вызова.")]));
  K.push(links([ xr("Алгоритм → §6.8","s_algo_peaks"), sep(), flink("src/optivibe/analysis/expected_peaks.py"), sep(), flink("src/optivibe/viz/dsp.py") ]));

  K.push(H(2,"7.13 Композит и модуляция возбуждения","s_impl_composite"));
  K.push(para([flink("src/optivibe/excitation/composite.py","excitation/composite.py"), R(" — источник, у которого "), R("нет своей физики сигнала", { bold: true }), R(": он вызывает те же зарегистрированные генераторы и складывает результаты поосевно.")]));
  K.push(...code([
    "def component_seed(scenario_seed: int | None, index: int) -> int | None:",
    '    """0 → сид сценария без изменений; i≥1 → SeedSequence([seed, TAG, i])."""',
    "",
    "class CompositeExcitationSource:",
    "    def generate(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:",
    "        # meta: суммарные rms/peak и покомпонентные rms — для предупреждения о FS",
  ]));
  K.push(para([R("Модуляция живёт в моделях конфигурации и в тональном генераторе: "), c("AmModulation"), R("/"), c("FmModulation"), R(" за союзом "), c("Modulation"), R(", опциональные "), c("SineSpec.modulation"), R(" и "), c("SineSpec.phase_rad"), R(", опциональный "), c("RandomSpec.seed"), R(". Все поля строго опт-ин: "), c("m = 0"), R(", "), c("β_FM = 0"), R(" и композит из одной компоненты дают "), R("побайтово тот же массив", { bold: true }), R(", что до-S-21 путь.")]));
  K.push(links([ xr("Алгоритм → §6.1","s_algo_composite"), sep(), flink("src/optivibe/excitation/composite.py"), sep(), flink("src/optivibe/excitation/tonal.py") ]));

  K.push(H(2,"7.14 Ввод измеренных параметров","s_impl_ingest"));
  K.push(para([flink("src/optivibe/io/characterization.py","io/characterization.py"), R(" — контракт измерения и ридеры по видам; "), flink("src/optivibe/io/ingest.py","io/ingest.py"), R(" — путь «измерение → конфиг» (§3.5).")]));
  K.push(...code([
    "# characterization.py",
    "class MeasuredParameter(BaseModel):   # frozen",
    "    name: str; value: float; u: float | None; method: str   # значение в СИ",
    "class Provenance(BaseModel):          # frozen",
    "    kind: str; sidecar: Path; data_file: Path | None; sha256: str",
    "    instrument: str; timestamp: str",
    "CHARACTERIZATION_REGISTRY: Registry[CharacterizationReader]",
    "    #  scalar · spectrum · rin_psd · ringdown · profile",
    "def load_characterization(sidecar_path: Path | str) -> CharacterizationResult: ...",
    "def resolve_sidecar_path(path: Path | str) -> Path:   # CSV ↔ YAML по общему stem",
    "",
    "# ingest.py",
    "PARAMETER_TARGETS: dict[str, tuple[str, str]]   # имя параметра → (блок, поле)",
    "def apply_measurements(system: SystemConfig,",
    "                       results: Sequence[CharacterizationResult], *,",
    "                       constants: Constants | None = None) -> IngestReport: ...",
    "def save_provenance(report: IngestReport, composition_path: Path) -> Path: ...",
  ]));
  K.push(para([R("Дисциплина, закодированная в "), c("apply_measurements"), R(", а не оставленная на оператора: правило GUM (без "), c("u"), R(" параметр в конфиг не пишется), анти-даблкаунт (измеренное "), R("замещает", { bold: true }), R(" модельное), семантика override для вычислимых полей, отсутствие конфиг-слота у "), c("f1_hz"), R(" и "), c("dop"), R(", конфликт при двух измерениях одного поля. Ключ "), c("PARAMETER_TARGETS"), R(" совпадает с именем конфиг-поля один в один — слоя переименований, который мог бы разъехаться, нет.")]));
  K.push(links([ xr("Продуктовое описание → §3.5","s_ingest"), sep(), flink("src/optivibe/io/characterization.py"), sep(), flink("src/optivibe/io/ingest.py") ]));

  K.push(H(2,"7.15 Стенд сравнения и потоковые воркеры","s_impl_compare"));
  K.push(para([flink("src/optivibe/analysis/compare.py","analysis/compare.py"), R(" — Qt-free ядро §3.7. Вердикт "), R("вычисляется", { bold: true }), R(", а карта применимости объявляется на "), R("все", { bold: true }), R(" поля модели (полнота закреплена тестом: новое поле не может появиться без тега):")]));
  K.push(...code([
    "DEFAULT_CHAIN = DspOptions()                  # верифицированная цепочка — только она",
    "ChainStatus = Literal['verified', 'experimental']",
    "CHAIN_APPLICABILITY: dict[str, Literal['batch','stream','both']]",
    "EXPERIMENT_FIELDS: tuple[str, ...]            # порядок строк панели",
    "",
    "def chain_status(options: DspOptions) -> ChainStatus: ...",
    "def chain_deltas(options: DspOptions) -> tuple[ChainDelta, ...]:   # что именно отклонилось",
    "def compare_chains(source: CompareInput, chains: Sequence[ChainSpec], *,",
    "                   name: str = 'comparison', constants=None) -> ComparisonResult:",
    "    # каждая цепочка исполняется НЕИЗМЕНЁННЫМ StandardDsp (17 §7)",
    "def input_from_scenario(...) -> CompareInput      # тот же шов, что у ScenarioSource",
    "def input_from_analyze_spec(...) -> CompareInput  # тот же шов, что у analyze_record",
    "def chain_provenance(...) / provenance_yaml(...)  # вердикт + дифф + версия/HEAD",
  ]));
  K.push(para([flink("src/optivibe/gui/workers/stream.py","gui/workers/stream.py"), R(" — источники и цикл живого режима, "), R("Qt-free", { bold: true }), R(" (проверено импортом в окружении без PySide6), поэтому при появлении второго потребителя переносится в ядро одним движением:")]));
  K.push(...code([
    "class StreamSource(Protocol): ...          # ScenarioSource | RecordSource",
    "@dataclass(frozen=True)",
    "class StreamFrame:",
    "    result: VibrationResult; warmed: bool",
    "    dropped_samples: int | None            # None в ускоренном режиме — НЕ 0",
    "    n_samples: int; elapsed_s: float; loops: int; seam: bool; paced: bool",
    "    source_label: str",
    "def default_block_size(fs: float, rate_hz: float) -> int: ...",
    "def run_stream(...)                        # часы и sleep инжектируются → тестируемо",
  ]));
  K.push(para([R("Виджеты — тонкие: "), flink("src/optivibe/gui/widgets/live_controls.py","gui/widgets/live_controls.py"), R(" (управление и строка провенанса; отдельный виджет "), R("нарочно", { italics: true }), R(" — это точка расширения, а не монолит внутри вида), "), flink("src/optivibe/gui/widgets/dsp_controls.py","gui/widgets/dsp_controls.py"), R(" (панель эксперимента — "), R("вид", { italics: true }), R(" на "), c("DspOptions"), R(": эмитит полезную нагрузку, ничего не запускает и не хранит своей истины), "), flink("src/optivibe/gui/widgets/compare_panel.py","gui/widgets/compare_panel.py"), R(" (вкладка сравнения). Расчёт сравнения идёт "), c("CompareJob"), R("'ом в существующем семействе воркеров, то есть вне UI-потока.")]));
  K.push(links([ xr("Продуктовое описание → §3.7","s_cmp"), sep(), xr("Живой режим → §3.6","s_live"), sep(), flink("src/optivibe/analysis/compare.py"), sep(), flink("src/optivibe/gui/workers/stream.py") ]));


  return K;
}
