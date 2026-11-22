# D램 커패시터 누설 억제를 위한 산소 차단 삽입층 소재 설계
**Design of Oxygen Barrier Interlayer Materials for Leakage Suppression in DRAM Capacitors**

DRAM 커패시터(TiN / ZrO₂ 계열 / TiN)에서 ZrO₂ ALD(O₃)와 후속 열처리 중 아래 TiN 전극이 산화되면
TiOₓNᵧ 계면층이 생겨 EOT가 늘고(C_s 감소) 누설이 늘어난다. 이 저장소는 TiN과 ZrO₂ 사이에 넣을
약 1 nm 산소 차단 삽입층(oxygen diffusion barrier interlayer)을 **Materials Project 스크리닝 + 범용 MLIP(M3GNet)**로 고른 계산 전체를 담는다.

- MLIP: `M3GNet-PES-MatPES-PBE-2025.2` (matgl 4.1.4), ASE + `PESCalculator(stress_unit="eV/A3")`
- DFT는 직접 하지 않았다. 검증은 MP(PBE) 값과 문헌 실험·DFT 값으로 했다.
- 모든 결론은 **같은 조건에서의 상대 순위**이다.

## 결과 요약
| 후보 | 성질 | O 이동 장벽 E_m (eV) | EOT penalty (nm, 1 nm 삽입층 전부 산화) | 비고 |
|---|---|---|---|---|
| TiN (기준) | 전도성 | 1.00 | 2.34 (실측, Miao 2026) | |
| AlN (기준, 문헌 선례) | 절연성 | 1.48 | 0.45 | max(자기 EOT, 산화물 EOT) |
| **HfN** | 전도성 | **2.11** | 0.23 | |
| **ZrN** | 전도성 | **2.04** | 0.23 | 산화물 = ZrO₂, TiN과 열응력 최소 |
| **TaN** (rocksalt) | 전도성 | 1.58 | 0.38 | 준안정상 |
| **NbN** (rocksalt) | 전도성 | 1.62 | 0.22 | 준안정상 |
| **TiC** | 전도성 | 1.25 | 0.09* | *TiN과 같은 Ti 산화물 → 신뢰도 낮음 |

EOT penalty = (전도성) PBR × 3.9 / k_oxide, (절연성) max(3.9/k_self, PBR × 3.9/k_oxide). k는 문헌 박막 값(Robertson 2004; Nb₂O₅ 41), PBR은 Pilling–Bedworth 부피비. 2026-11-22 점검에서 수정한 내용은 `01_mp_funnel/조건변경이력.md` 참고.

전체 18종 표: [`04_summary/summary_table.csv`](04_summary/summary_table.csv) · 선정 지도: [`04_summary/selection_map.png`](04_summary/selection_map.png) · 그림: [`04_summary/figures/`](04_summary/figures)

## 계산 단계
| 폴더 | 단계 | 내용 |
|---|---|---|
| `01_mp_funnel/` | ① MP 깔때기 | N0 449 조성 → N1 297 (E_hull<0.05 eV/atom 또는 ICSD) → N2 223 (`InterfacialReactivity`, TiN·ZrO₂ > −0.1 eV/atom) → N3 215 (EOT penalty ≤ TiN 실측 2.34 nm) → N4 ALD DB 130 → 2성분 대표 17종. thermo type `GGA_GGA+U` 통일 |
| `02_relax_elastic/` | ② 이완·탄성 | FIRE + FrechetCellFilter, ±0.5/1% strain → C_ij → VRH B, G, E, B/G |
| `03_neb/` | ③ NEB (주 지표) | 216–321원자, O_i(최저 자리 탐색 + 대칭 동등 자리) / O_X+V_X, 7 images, 2단계 CI-NEB, 발산 시 중간점 2구간 |
| `05_cte/` | ⑤ CTE | phonopy QHA, 부피 7개, Vinet EOS (상대 비교 전용) |
| `06_ysz_md/` | ④ 방법 검증 | 8YSZ 316원자, 800/1000/1200 K × seed 2, 10 ps NPT + 100 ps NVT → MSD → D → Arrhenius |
| `04_summary/` | 종합 | 결과표, 선정 지도, 발표 그림 |

조건을 바꾼 모든 이력과 이유: [`01_mp_funnel/조건변경이력.md`](01_mp_funnel/조건변경이력.md)

## 재현
```bash
conda create -n m3gnet python=3.11 -y && conda activate m3gnet
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r laptop_handoff/requirements_m3gnet.txt
python laptop_handoff/check_env.py          # TiN a ≈ 4.249 Å
export MP_API_KEY=...                       # 키는 저장소에 넣지 않는다
python 01_mp_funnel/funnel.py && python 01_mp_funnel/n3_table.py
python 02_relax_elastic/relax_elastic.py TiN AlN ...
python 03_neb/neb.py TiN N                  # NAME ANION [Oi] [ON_VN]
python 05_cte/cte.py TiN ...
python 06_ysz_md/md.py 1000 1 && python 06_ysz_md/analyse.py
python 04_summary/summary.py && python 04_summary/figures.py
```

## 데이터 출처
- Materials Project (Jain et al., APL Mater. 1, 011002, 2013), mp-api, thermo type GGA_GGA+U
- Atomic Limits ALD process database, DOI 10.6100/alddatabase (CC BY 4.0), 2026-11-20 조회 → `01_mp_funnel/ald_database_2026-11-20.json`
- TiN 기준 EOT 증가: Miao et al., Appl. Phys. Lett. 128, 022905 (2026), Fig. 3a
- YSZ 실험 E_a: Kilo et al., Phys. Chem. Chem. Phys. 5, 2219 (2003), 초록: 0.8–1.0 eV
- 문헌 목록: [`papers/00_논문목록.md`](papers/00_논문목록.md)

## 한계 (주장하지 않는 것)
1. 누설 전류·일함수·밴드 정렬은 계산하지 않았다(MLIP는 전자를 다루지 않음).
2. 1 nm 계면을 벌크 결정으로 근사했다. 결정립계 확산, 산화물 scale 안 확산(Deal–Grove)은 다루지 않았다.
3. 결함은 중성만 다룬다. 메커니즘마다 경로 하나만 계산했다.
4. 산화물 ALD 막은 비정질이지만 결정상으로 계산했다. M3GNet은 HfO₂의 O–O dimer를 재현하지 못했다.
5. EOT penalty의 k는 문헌 박막 값(일부 상은 MP DFPT, 없으면 k = 3.9 상한)이고, 산화물 두께는 결정 부피비(PBR)로 근사했다.
6. CTE: rocksalt TaN·NbN은 팽창 시 연성 모드, ZrO₂(정방·단사)는 M3GNet에서 허수 모드가 나와 QHA를 적용하지 못했다.
7. YSZ 검증에서 M3GNet의 O 확산 E_a는 0.42 ± 0.06 eV(Langevin 0.01/fs), 0.45 ± 0.17 eV(0.001/fs)로 실험 0.85–1.1 eV(Vendrell et al., J. Electrochem. Soc. 2018)보다 낮다. thermostat를 바꿔도 같으므로 uMLIP의 PES softening(Deng et al., npj Comput. Mater. 11, 9 (2025)) 방향과 일치 → 장벽은 절대값이 아니라 순위로만 쓴다. 그림: `04_summary/figures/fig6_ysz_arrhenius.png`.
8. ⑦ 저장벽 MD(1 seed × 50 ps × 1000/1200/1400 K)는 NEB가 낮다고 한 산화물에서 O_i가 실제로 빨리 움직이는지 확인하는 정성 검증이다. Al₂O₃·HfO₂는 MD E_a(0.10 ± 0.09, 0.17 ± 0.09 eV)가 NEB(0.32, 0.27 eV) 이하라서 "산소 차단이 약하다"는 결론이 유지된다. Si₃N₄는 1000–1200 K에서 O_i가 거의 움직이지 않았고, Nb₂O₅·SiO₂는 host 구조가 변해 비교하지 않았다. 분석: `09_lowbarrier_md/analyse_defect_v2.py`.
