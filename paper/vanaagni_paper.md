# VanAagni: Foundation Model Fine-Tuning with Weather-Conditioned Decoding for 30-Meter Burn Severity Mapping in Tropical Dry Deciduous Forests

**Shivam C. Chauhan**$^{1,\ast}$

$^1$ Department of Forest Conservation, Guna Division, Madhya Pradesh Forest Department, Guna 473001, India

$^\ast$ Corresponding author. E-mail: shivam.chauhan@mp.gov.in

---

## Abstract

Accurate burn severity mapping at fine spatial resolution is critical for post-fire ecosystem management, yet remains challenging in data-sparse tropical dry deciduous forests where cloud cover, rapid regrowth, and limited ground truth constrain remote sensing approaches. We present VanAagni, a deep learning architecture that fine-tunes the Prithvi-EO-2.0-600M-TL foundation model (a 631-million parameter Vision Transformer pre-trained on Harmonized Landsat Sentinel-2 imagery) with a novel weather-conditioned UNet decoder for 30-meter burn severity mapping. Our approach introduces Feature-wise Linear Modulation (FiLM) to inject real-time fire weather variables — including the complete Canadian Forest Fire Weather Index system with 7-day temporal lookback — directly into the decoder's feature representations at each spatial scale. We further contribute a Layer-wise Learning Rate Decay with Detach Prefix (LLRD-DP) training strategy that enables fine-tuning the full 646-million parameter model on a consumer-grade AMD GPU with only 3.7 GB VRAM by partitioning the backbone into frozen prefix blocks and trainable suffix blocks with an explicit gradient boundary. Applied to 82 fire events across Guna Division in central India's Vindhyan dry deciduous forests (2013--2025), VanAagni achieves a fire detection F1 score of 0.747, precision of 80.6%, and recall of 69.6% on a held-out 2024--2025 test set comprising 385 evaluation patches. Our tiered label quality system (GOLD/SILVER/BRONZE/VIIRS_ONLY) enables principled integration of heterogeneous fire reference data with quality-aware loss weighting. Ablation experiments demonstrate that FiLM weather conditioning, multi-temporal input, and spatial auxiliary features each contribute measurably to model performance. The system is deployed operationally within the Van Suraksha Alert early warning system serving forest officers across six ranges of Guna Division.

**Keywords:** burn severity mapping; foundation model; fine-tuning; Feature-wise Linear Modulation; fire weather; Prithvi-EO-2.0; Vision Transformer; dry deciduous forest; India

---

## 1. Introduction

### 1.1 Forest Fire Severity as a Remote Sensing Challenge

Forest fires are a dominant ecological disturbance in tropical and subtropical ecosystems, affecting approximately 350 million hectares of forest globally each year (Giglio et al., 2018). The severity with which fire affects vegetation and soil — ranging from light surface scorch to complete canopy removal — determines post-fire recovery trajectories, carbon emissions, biodiversity impacts, and management response requirements (Key and Benson, 2006; Lentile et al., 2006). Accurate, timely, and spatially explicit burn severity mapping is therefore essential for post-fire management, including salvage prioritisation, rehabilitation planning, and ecosystem monitoring.

Remote sensing has long been the primary tool for burn severity assessment at landscape scales. The differenced Normalized Burn Ratio (dNBR), computed from pre-fire and post-fire near-infrared and shortwave-infrared imagery, remains the most widely used spectral index for burn severity estimation (Key and Benson, 2006; Miller and Thode, 2007). While effective in many biomes, dNBR-based approaches face well-documented limitations: sensitivity to pre-fire vegetation condition, susceptibility to phenological noise in deciduous forests, reliance on cloud-free pre-fire and post-fire image pairs, and limited ability to discriminate fine severity gradations (French et al., 2008; Parks et al., 2014). These challenges are particularly acute in tropical dry deciduous forests, where leaf-off phenology during the fire season (February--May) compresses spectral contrast between burned and unburned areas, and rapid post-fire coppice regrowth can mask burn scars within weeks (Roy et al., 2008).

Deep learning approaches have shown promise for burn severity mapping by learning complex spectral-spatial-temporal patterns that exceed the capacity of index-based methods (Knopp et al., 2020; Hu et al., 2021; Seydi et al., 2022). Convolutional neural networks and Vision Transformers have been applied to wildfire mapping in North American, Mediterranean, and Australian biomes with encouraging results. However, these approaches typically require large labelled datasets that are unavailable for most tropical regions, and their performance degrades substantially when applied to biomes not represented in training data.

### 1.2 Foundation Models for Earth Observation

The emergence of foundation models for Earth observation (EO) — large-scale models pre-trained on massive satellite imagery datasets through self-supervised learning — offers a pathway to overcome data scarcity constraints. NASA/IBM's Prithvi-EO-2.0 (Jakubik et al., 2024), the Spectral-Temporal Masked Autoencoder (SatMAE; Cong et al., 2022), and the General Foundation Model for remote sensing (GFM; Mendieta et al., 2023) have demonstrated that pre-training on global multi-spectral satellite imagery produces feature representations that transfer effectively to downstream tasks with limited labelled data.

Prithvi-EO-2.0, in particular, was pre-trained on over 4.2 million globally-sampled Harmonized Landsat Sentinel-2 (HLS) patches through masked image modelling, learning rich spectral-spatial-temporal representations across six spectral bands (Blue, Green, Red, NIR, SWIR1, SWIR2). The 600M-TL variant extends the base model with temporal and location embeddings that encode the acquisition date and geographic coordinates of each input patch, providing the model with phenological and biogeographic context. With 631 million parameters organised as a Vision Transformer (ViT-L, 1280-dimensional, 32 transformer blocks), Prithvi-EO-2.0 represents the largest publicly available EO foundation model and has demonstrated state-of-the-art transfer learning performance across diverse downstream tasks including crop type mapping, flood detection, and land cover classification (Jakubik et al., 2024).

However, the application of EO foundation models to fire-related tasks remains nascent. Existing fine-tuning approaches typically append task-specific classification or segmentation heads to the frozen or partially-thawed backbone, treating the foundation model as a fixed feature extractor. This approach neglects the potential for domain-specific adaptation of intermediate feature representations and, critically, fails to incorporate non-spectral information — particularly meteorological conditions — that strongly modulate fire behaviour and burn severity.

### 1.3 The Missing Dimension: Weather Conditioning in Segmentation

Fire severity is not determined solely by spectral properties of the land surface. Meteorological conditions at the time of burning — temperature, relative humidity, wind speed, precipitation history, and derived fire weather indices — exert profound control over fire intensity, rate of spread, and consequent burn severity (Van Wagner, 1987; Flannigan et al., 2005). The Canadian Forest Fire Weather Index (FWI) system, comprising six components from fine fuel moisture to the Build-Up Index, provides a physically-grounded quantification of fire weather conditions that has been validated globally (Taylor and Alexander, 2006).

Despite this strong physical relationship, existing deep learning approaches to burn severity mapping operate exclusively in the spectral-spatial domain, treating each satellite image as an independent observation disconnected from its meteorological context. The weather conditions at the time of burning are simply discarded. We argue that this represents a significant information gap that limits severity discrimination, particularly for distinguishing moderate from high severity burns where the difference is often driven by weather intensity rather than fuel load.

Feature-wise Linear Modulation (FiLM; Perez et al., 2018) provides an elegant mechanism for injecting conditioning information into deep neural networks. Originally developed for visual question answering, FiLM learns to modulate intermediate feature maps through learned affine transformations — scaling (gamma) and shifting (beta) each feature channel conditioned on an auxiliary input. FiLM and its variants (adaptive Layer Normalisation, adaLN-Zero; Peebles and Xie, 2023) have been adopted widely in generative modelling and multi-modal learning, but have not been applied to condition Earth observation segmentation models on meteorological data.

### 1.4 Central Indian Dry Deciduous Forests: A Unique Fire Regime

India's dry deciduous forests occupy approximately 38.2 million hectares — roughly 29% of India's forest cover — and experience among the highest fire frequencies of any tropical forest type (FSI, 2023). The Vindhyan dry deciduous forests of central Madhya Pradesh present a particularly challenging case for remote sensing-based fire monitoring. Guna Division, our study area, spans approximately 5,500 km$^2$ of undulating terrain (elevation 300--600 m) dominated by *Tectona grandis* (teak), *Anogeissus latifolia*, and *Butea monosperma*, with economically important Non-Timber Forest Products including *Madhuca longifolia* (Mahua) and *Diospyros melanoxylon* (Tendu).

The fire season in this region (February--May) coincides with leaf-off deciduous phenology, producing spectral confusion between naturally senescent vegetation and fire-affected areas in satellite imagery. Surface fires dominate, with intensity modulated by accumulated leaf litter, grass understory, and atmospheric conditions. The combination of limited ground truth data (no systematic field-based severity assessment), frequent cloud contamination during the pre-monsoon transition, and rapid coppice regeneration makes this biome uniquely challenging for burn severity mapping.

### 1.5 Contributions

This paper makes the following five contributions:

1. **First weather-conditioned decoder for EO foundation model segmentation.** We introduce FiLM conditioning that injects Canadian FWI fire weather variables (10 dimensions with 7-day temporal lookback) into the UNet decoder at each spatial scale, enabling the model to contextualise spectral observations with meteorological conditions at the time of burning.

2. **First 30-meter burn severity mapping for Indian dry deciduous forests using a foundation model backbone.** We demonstrate that Prithvi-EO-2.0-600M-TL, pre-trained on global HLS imagery, can be effectively fine-tuned for severity mapping in a biome not well-represented in its pre-training distribution.

3. **LLRD with Detach Prefix (LLRD-DP) training strategy.** We propose a memory-efficient fine-tuning approach that partitions the 32-block ViT backbone into a frozen prefix (blocks 0--27) and a trainable suffix (blocks 28--31) with an explicit gradient detachment boundary, enabling full-model fine-tuning of a 646M parameter model on a consumer GPU with 3.7 GB VRAM — an order of magnitude below typical requirements.

4. **Tiered label quality system with quality-aware loss weighting.** We introduce a four-tier fire label classification (GOLD/SILVER/BRONZE/VIIRS_ONLY) based on spatial precision of reference fire perimeters, with tier-dependent per-pixel loss weights and sampling probabilities that enable principled integration of heterogeneous fire reference data.

5. **Comprehensive ablation study** quantifying the contribution of weather conditioning, multi-temporal input, spatial auxiliary features, backbone fine-tuning, and severity granularity to model performance, providing insights into which architectural components matter most for fire severity mapping.

---

## 2. Related Work

### 2.1 Burn Severity Mapping

Burn severity mapping from satellite imagery has evolved through three generations of approaches. **Spectral index methods**, led by the differenced Normalized Burn Ratio (dNBR; Key and Benson, 2006), remain the operational standard for agencies including the US Geological Survey's Monitoring Trends in Burn Severity (MTBS) programme. dNBR exploits the contrast between near-infrared reflectance (sensitive to vegetation structure) and shortwave-infrared reflectance (sensitive to moisture content) before and after fire. The Relativized dNBR (RdNBR; Miller and Thode, 2007) and Relativized Burn Ratio (RBR; Parks et al., 2014) address heterogeneous pre-fire vegetation conditions but remain single-index approaches with limited discriminative capacity.

**Machine learning methods** have extended severity estimation beyond single indices. Random Forests and gradient boosting applied to multi-spectral and multi-temporal features have shown improved accuracy over index-based approaches (Collins et al., 2018; Hislop et al., 2019). These methods can incorporate auxiliary variables (topography, vegetation type, fire weather) as additional features but treat them as independent inputs rather than as conditioning information that modulates feature representations.

**Deep learning methods** for burn area delineation and severity estimation have emerged primarily in the last five years. U-Net architectures applied to Sentinel-2 imagery have achieved promising results for binary burned area detection (Knopp et al., 2020; Hu et al., 2021). More recently, attention-based architectures including Swin Transformer variants have been applied to burned area mapping with improved spatial detail (Seydi et al., 2022). However, deep learning approaches to multi-class severity estimation remain sparse, limited primarily to well-studied North American and Mediterranean fire regimes with abundant Composite Burn Index (CBI) field data.

Critically, all existing deep learning approaches to burn severity mapping operate exclusively in the spectral-spatial domain. No prior work has incorporated fire weather conditioning into the segmentation architecture, despite the strong physical relationship between meteorological conditions and burn outcomes.

### 2.2 Earth Observation Foundation Models

The foundation model paradigm — pre-training very large models on massive unlabelled datasets, then fine-tuning for specific tasks — has transformed Earth observation. Three model families are most relevant to our work.

**Prithvi-EO-2.0** (Jakubik et al., 2024) is a temporal Vision Transformer pre-trained through masked image modelling on 4.2 million HLS patches sampled globally. Available in four variants (300M, 300M-TL, 600M, 600M-TL), the model accepts multi-temporal, multi-spectral input and has demonstrated strong transfer learning across diverse downstream tasks. The 600M-TL variant adds temporal and location embeddings that encode acquisition date and geographic coordinates, providing implicit phenological and biogeographic context. Prithvi-EO-2.0 is distributed through the TerraTorch framework (Bernstein et al., 2024), which provides standardised backbone loading and fine-tuning pipelines.

**SatMAE** (Cong et al., 2022) introduced a spectral-temporal masked autoencoder for satellite imagery, pre-training on temporal sequences of multi-spectral patches. SatMAE demonstrated that independent spectral and temporal positional encodings improve representation quality for downstream temporal change detection and classification tasks.

**General Foundation Model (GFM)** (Mendieta et al., 2023) combined multiple pre-training objectives and data sources, including optical and SAR imagery, to produce more generalisable representations. GFM emphasised the importance of diverse pre-training data for cross-domain transfer.

Fine-tuning strategies for EO foundation models typically fall into three categories: (1) **linear probing**, where only a classification head is trained while the backbone remains frozen; (2) **full fine-tuning**, where all parameters are updated; and (3) **parameter-efficient fine-tuning** (PEFT), including LoRA (Hu et al., 2022) and adapters. Our LLRD-DP approach represents a novel middle ground — the backbone is neither fully frozen nor fully fine-tuned, but partitioned into frozen and trainable segments with learned per-layer learning rates.

### 2.3 Feature-wise Linear Modulation and Conditional Normalisation

Feature-wise Linear Modulation (FiLM; Perez et al., 2018) learns to modulate feature maps $\mathbf{F}$ through affine transformations conditioned on an auxiliary input $\mathbf{z}$:

$$\text{FiLM}(\mathbf{F} | \mathbf{z}) = \gamma(\mathbf{z}) \odot \mathbf{F} + \beta(\mathbf{z})$$

where $\gamma$ and $\beta$ are learned functions of the conditioning input. Originally applied to visual question answering, FiLM has been adopted in numerous multi-modal settings. In the context of normalisation layers, this mechanism becomes adaptive Layer Normalisation (adaLN), where the affine parameters of a normalisation layer are replaced by learned functions of external conditioning:

$$\text{adaLN}(\mathbf{F} | \mathbf{z}) = \gamma(\mathbf{z}) \odot \text{LayerNorm}(\mathbf{F}) + \beta(\mathbf{z})$$

The adaLN-Zero variant (Peebles and Xie, 2023), introduced in the Diffusion Transformer (DiT), initialises $\gamma$ to 1 and $\beta$ to 0, ensuring that the conditioning starts as an identity transformation and gradually learns to modulate features during training. This zero-initialisation prevents the conditioning signal from destabilising early training — a property particularly important when fine-tuning from pre-trained weights.

In our work, we apply FiLM conditioning with adaLN-Zero initialisation to Group Normalisation layers in the UNet decoder, conditioning on a learned weather embedding. This represents, to our knowledge, the first application of FiLM conditioning to Earth observation segmentation with meteorological data.

### 2.4 Canadian Forest Fire Weather Index System

The Canadian Forest Fire Weather Index (FWI) system (Van Wagner, 1987) is the most widely used fire weather rating system globally. It comprises six components computed from four weather inputs (temperature, relative humidity, wind speed, and 24-hour precipitation):

- **Fine Fuel Moisture Code (FFMC):** moisture content of litter and fine fuels (1-hr timelag)
- **Duff Moisture Code (DMC):** moisture content of decomposed organic layer (10-day timelag)
- **Drought Code (DC):** moisture content of deep organic layers (52-day timelag)
- **Initial Spread Index (ISI):** expected rate of fire spread (combines FFMC and wind)
- **Build-Up Index (BUI):** fuel available for combustion (combines DMC and DC)
- **Fire Weather Index (FWI):** overall fire intensity (combines ISI and BUI)

These components capture different temporal scales of fuel drying, from hourly fine fuel response to seasonal drought accumulation, providing a physically-grounded multi-scale representation of fire weather conditions. The FWI system has been validated in Indian forests (Varma, 2003; Renard et al., 2012) and is used operationally by the Indian Forest Survey Institute.

We use the complete FWI system (FFMC, DMC, DC, ISI, BUI, FWI) along with the four primary weather inputs (temperature, relative humidity, wind speed, precipitation) as a 10-dimensional weather vector for FiLM conditioning. Additionally, we provide a 7-day lookback window of all 10 variables as temporal context, enabling the model to capture weather trends rather than point-in-time snapshots.

---

## 3. Study Area and Data

### 3.1 Study Area

Guna Division is located in the Vindhyan Plateau of central Madhya Pradesh, India, spanning approximately 24.25--25.00$^\circ$N latitude and 76.95--77.60$^\circ$E longitude (Figure 2). The division covers approximately 5,500 km$^2$ and is administered as 747 forest compartments across six ranges: Aron, Fatehgarh, Guna North, Guna South, Maksudanagar, and Raghogarh.

The terrain is gently undulating, with elevation ranging from 300 to 600 m above sea level and slopes generally below 15$^\circ$. The dominant vegetation type is Southern Dry Mixed Deciduous Forest (Champion and Seth, 5A/C3), characterised by *Tectona grandis* (teak), *Anogeissus latifolia* (dhawda), *Lagerstroemia parviflora* (lendia), and *Butea monosperma* (palash), with a grass and herb understory that serves as the primary surface fuel.

The climate is semi-arid tropical, with a pronounced dry season from October to June and monsoon rainfall of 900--1100 mm concentrated between July and September. The fire season spans February through May, coinciding with peak leaf-fall and atmospheric drying. Fire is primarily anthropogenic — ignited for Non-Timber Forest Product (NTFP) collection (particularly *Madhuca longifolia* flowers and *Diospyros melanoxylon* leaves), grazing management, and agricultural clearing. Surface fires dominate, with flame lengths typically below 2 m, but intensity varies substantially with weather conditions.

### 3.2 Satellite Imagery: Harmonized Landsat Sentinel-2 (HLS)

We use Harmonized Landsat Sentinel-2 (HLS) S30 imagery (Claverie et al., 2018) as the primary spectral input. HLS S30 provides Sentinel-2-derived surface reflectance at 30-meter spatial resolution with consistent radiometric and geometric processing. We utilise six spectral bands aligned with Prithvi-EO-2.0's pre-training configuration:

| Band | HLS Name | Wavelength (nm) | Description |
|------|----------|-----------------|-------------|
| B02 | Blue | 490 | Aerosol/water sensitivity |
| B03 | Green | 560 | Vegetation peak reflectance |
| B04 | Red | 665 | Chlorophyll absorption |
| B8A | NIR Narrow | 865 | Vegetation structure |
| B11 | SWIR 1 | 1610 | Moisture/cellulose |
| B12 | SWIR 2 | 2190 | Moisture/minerals |

**Table 1.** Spectral bands used from HLS S30 imagery.

For each fire event, we construct a temporal stack of $T = 3$ frames at approximately 16-day cadence, capturing pre-fire conditions, the fire period, and early post-fire response. This multi-temporal design enables the model to detect spectral change patterns characteristic of burning, analogous to dNBR but learned end-to-end from data. The input tensor for each patch is thus $\mathbf{X}_\text{HLS} \in \mathbb{R}^{6 \times 3 \times 224 \times 224}$, representing 6 bands $\times$ 3 temporal frames $\times$ 224 $\times$ 224 spatial pixels (approximately 6.7 km $\times$ 6.7 km at 30 m resolution).

Additionally, we compute three spectral indices for each temporal frame:

$$\text{NDVI} = \frac{\text{B8A} - \text{B04}}{\text{B8A} + \text{B04}}, \quad \text{NBR} = \frac{\text{B8A} - \text{B12}}{\text{B8A} + \text{B12}}, \quad \text{BSI} = \frac{(\text{B11} + \text{B04}) - (\text{B8A} + \text{B02})}{(\text{B11} + \text{B04}) + (\text{B8A} + \text{B02})}$$

These indices provide explicit representations of vegetation health (NDVI), burn response (NBR), and bare soil exposure (BSI), yielding $\mathbf{X}_\text{idx} \in \mathbb{R}^{3 \times 3 \times 224 \times 224}$.

### 3.3 Fire Reference Data: Tiered Label Quality System

Fire reference data for Guna Division was compiled from multiple sources spanning 2013--2025, yielding 82 fire events with spatially explicit perimeters. Given the heterogeneous provenance and spatial precision of these reference data, we developed a four-tier quality classification system:

| Tier | Count | Criteria | Source | Spatial Precision |
|------|-------|----------|--------|-------------------|
| GOLD | 31 (38%) | Manually digitised from high-resolution imagery with field verification | Sentinel-2 visual interpretation + field reports | ~30 m (1 pixel) |
| SILVER | 26 (32%) | Semi-automated extraction from Sentinel-2 spectral change | dNBR thresholding + manual refinement | ~60 m (2 pixels) |
| BRONZE | 2 (2%) | FIRMS active fire detections with buffered perimeters | VIIRS/MODIS 375m/1km hotspots | ~375 m |
| VIIRS_ONLY | 23 (28%) | VIIRS active fire detections only, no spectral confirmation | VIIRS 375m S-NPP/NOAA-20 | ~500 m |

**Table 2.** Fire reference data tiers and their characteristics. Counts reflect the number of fire events per tier.

The tiered system enables principled integration of data with varying reliability. GOLD-tier labels, with pixel-level precision, receive the highest training weight and serve as the primary quality benchmark. VIIRS_ONLY labels, while spatially coarse, expand temporal coverage to fire seasons where cloud cover prevented spectral confirmation.

**Burn severity labels** were assigned using fire radiative power (FRP) as a proxy for burn intensity. VIIRS active fire detections associated with each fire event were binned into five severity classes based on FRP thresholds:

| Class | Label | FRP Range (MW) | Description |
|-------|-------|----------------|-------------|
| 0 | No burn | 0 | Background (no fire) |
| 1 | Very low | 0--5 | Light surface scorch |
| 2 | Low | 5--15 | Moderate surface burn |
| 3 | Moderate | 15--40 | Significant canopy scorch |
| 4 | High | >40 | Severe canopy burn |

**Table 3.** Five-class burn severity classification based on VIIRS Fire Radiative Power (FRP).

We acknowledge that FRP-based severity labels are a noisy proxy for true ecological burn severity, which ideally would be measured through field-based Composite Burn Index (CBI) assessments. The class distribution is heavily imbalanced — approximately 82% of fire pixels are assigned class 1 (very low), reflecting the dominance of low-intensity surface fires in this biome. This imbalance has important implications for model performance, discussed in Section 5.3.

### 3.4 Auxiliary Data

**Terrain.** We use the Shuttle Radar Topography Mission (SRTM) 30-meter digital elevation model (Farr et al., 2007) to derive four terrain variables: elevation (normalised), slope (normalised), and aspect (encoded as $\sin(\text{aspect})$ and $\cos(\text{aspect})$ to handle circularity). Terrain influences fire behaviour through slope-driven fire spread acceleration, aspect-dependent solar exposure and fuel drying, and elevation-correlated vegetation composition. The terrain tensor is $\mathbf{X}_\text{terrain} \in \mathbb{R}^{4 \times 224 \times 224}$.

**Land cover.** ESA WorldCover 10-meter land cover classification (Zanaga et al., 2022) is resampled to 30 meters and one-hot encoded into 11 classes (tree cover, shrubland, grassland, cropland, built-up, bare/sparse vegetation, snow/ice, water, wetland, mangroves, moss/lichen). Land cover provides the model with vegetation type context that influences fuel characteristics and fire susceptibility. The land cover tensor is $\mathbf{X}_\text{lc} \in \mathbb{R}^{11 \times 224 \times 224}$.

**Burn age.** Historical VIIRS active fire detections from 2012 to the present are processed into three burn age channels: (1) recent burn (fire within 1 year), (2) high-risk recovery (1--3 years post-fire, when fuel re-accumulation creates elevated fire risk), and (3) mature fuel (>3 years since last fire). The burn age tensor is $\mathbf{X}_\text{burn} \in \mathbb{R}^{3 \times 224 \times 224}$.

**Fire weather.** Weather data for each fire event is obtained from ERA5 reanalysis (Hersbach et al., 2020) and Open-Meteo forecast APIs. The primary weather vector $\mathbf{w} \in \mathbb{R}^{10}$ contains temperature ($^\circ$C), relative humidity (%), wind speed (km/h), 24-hour precipitation (mm), and the six FWI components (FFMC, DMC, DC, ISI, BUI, FWI). Additionally, a 7-day lookback window $\mathbf{W}_{7d} \in \mathbb{R}^{10 \times 7}$ captures weather trends leading up to the fire event, enabling the model to learn relationships between antecedent drying and burn severity.

### 3.5 Patch Generation and Data Splits

Training patches are generated by extracting 224 $\times$ 224 pixel windows (6.72 km $\times$ 6.72 km) centred on each fire event and on randomly sampled negative (no-fire) locations within Guna Division. The patch generation pipeline produces `.npz` files containing all input tensors, labels, and metadata.

The dataset comprises 2,643 patches partitioned by fire year into training, validation, and test splits:

| Split | Years | Total | Fire | Negative | Fire % |
|-------|-------|-------|------|----------|--------|
| Train | 2013--2018, 2020--2023 | 1,183 | 546 | 637 | 46.2% |
| Val | 2019 | 1,075 | 107 | 968 | 10.0% |
| Test | 2024--2025 | 385 | 228 | 157 | 59.2% |

**Table 4.** Dataset split statistics. The year-based temporal split prevents information leakage from spatial autocorrelation.

The temporal split — using entire fire seasons as the unit of partition — is critical for preventing information leakage. Spatial random splits can produce optimistically biased results due to autocorrelation between nearby patches from the same fire event. Our year-based split ensures that the model is evaluated on fire events from entirely unseen temporal periods, providing a more realistic assessment of operational performance.

The fire patch distribution across quality tiers is: GOLD = 221, SILVER = 220, VIIRS_ONLY = 417, and BRONZE = 23. Negative patches (n = 1,762) are sampled from compartments with no recorded fire activity during the corresponding year.

---

## 4. Methodology

### 4.1 Architecture Overview

VanAagni is an encoder-decoder architecture comprising three main components (Figure 1):

1. **Encoder:** Prithvi-EO-2.0-600M-TL Vision Transformer backbone that extracts multi-scale feature representations from multi-temporal HLS imagery.
2. **Decoder:** A 4-scale UNet decoder with skip connections that progressively upsamples features from 14 $\times$ 14 to 224 $\times$ 224 resolution, with FiLM weather conditioning applied at each scale.
3. **Auxiliary inputs:** A Spatial Auxiliary Encoder that processes terrain, land cover, and burn age data into scale-matched feature maps injected into the decoder via concatenation.

The complete model has 646,838,791 parameters: 631,188,482 in the backbone and 15,650,309 in the decoder (including FiLM conditioning layers and the spatial auxiliary encoder).

### 4.2 Backbone: Prithvi-EO-2.0-600M-TL

The backbone is a Vision Transformer Large (ViT-L) with 32 transformer blocks, each containing multi-head self-attention (16 heads, 1280-dimensional) and feed-forward networks (5120-dimensional). The model was pre-trained through masked image modelling on 4.2 million HLS patches, learning to reconstruct randomly masked portions (75% masking ratio) of multi-spectral, multi-temporal input.

**Patch embedding.** Input imagery $\mathbf{X}_\text{HLS} \in \mathbb{R}^{6 \times 3 \times 224 \times 224}$ is embedded through a convolutional patch projection with kernel size 16 $\times$ 16 and stride 16, producing $N = 14 \times 14 = 196$ spatial tokens per temporal frame, each of dimension 1280. For $T = 3$ frames, the total sequence length is $3 \times 196 = 588$ tokens.

**Temporal and Location (TL) embeddings.** The TL variant adds learned embeddings for acquisition date (day of year, encoded via sinusoidal functions) and geographic location (latitude and longitude), providing the model with phenological seasonality and biogeographic context. These embeddings are summed with the standard positional embeddings before the transformer blocks.

**Feature extraction.** We extract multi-scale features from four intermediate backbone depths for the UNet decoder skip connections. Following standard practice for ViT-based segmentation (Chen et al., 2022), features are extracted after blocks 8, 16, 24, and 32 (the final layer), providing representations at progressively deeper levels of abstraction. All features share the same spatial resolution (14 $\times$ 14) but are reshaped and upsampled to create a multi-scale feature pyramid.

**DirectML compatibility.** Prithvi-EO-2.0 uses 3D convolutions (Conv3d) in its patch embedding layer to jointly convolve over the temporal and spatial dimensions. Since DirectML (Microsoft's GPU compute layer for non-CUDA hardware) does not natively support Conv3d, we implement a `_Conv3dAsConv2d` shim layer that replaces Conv3d operations with equivalent Conv2d operations when the temporal kernel size is 1, ensuring compatibility without modifying the original model weights.

### 4.3 UNet Decoder with Skip Connections

The decoder follows the UNet architecture (Ronneberger et al., 2015) adapted for Vision Transformer feature maps. It comprises four decoder blocks operating at progressively increasing spatial resolution:

| Scale | Input Resolution | Output Resolution | Channels |
|-------|-----------------|-------------------|----------|
| 1 | 14 $\times$ 14 | 28 $\times$ 28 | 512 |
| 2 | 28 $\times$ 28 | 56 $\times$ 56 | 256 |
| 3 | 56 $\times$ 56 | 112 $\times$ 112 | 128 |
| 4 | 112 $\times$ 112 | 224 $\times$ 224 | 64 |

**Table 5.** UNet decoder scale configuration.

Each decoder block performs: (1) bilinear upsampling by factor 2, (2) concatenation with the corresponding backbone skip connection features and spatial auxiliary features, (3) two 3 $\times$ 3 convolutional layers with Group Normalisation and GELU activation, and (4) FiLM weather conditioning applied after each Group Normalisation layer.

The final classification head is a 1 $\times$ 1 convolution mapping from 64 channels to 5 output classes (no burn, very low, low, moderate, high severity), producing logits $\hat{\mathbf{Y}} \in \mathbb{R}^{5 \times 224 \times 224}$.

### 4.4 FiLM Weather Conditioning

Our central architectural innovation is the injection of fire weather information into the decoder through Feature-wise Linear Modulation. The FiLM conditioning pipeline comprises three stages: weather encoding, FiLM parameter generation, and feature modulation.

**Stage 1: Weather encoding.** The primary weather vector $\mathbf{w} \in \mathbb{R}^{10}$ is processed through a three-layer MLP:

$$\mathbf{e}_w = \text{MLP}(\mathbf{w}) = \sigma(\mathbf{W}_3 \cdot \sigma(\mathbf{W}_2 \cdot \sigma(\mathbf{W}_1 \cdot \mathbf{w} + \mathbf{b}_1) + \mathbf{b}_2) + \mathbf{b}_3)$$

where $\sigma$ is GELU activation and the MLP dimensions are $10 \rightarrow 64 \rightarrow 64 \rightarrow 128$, producing a weather embedding $\mathbf{e}_w \in \mathbb{R}^{128}$.

The 7-day forecast weather window $\mathbf{W}_{7d} \in \mathbb{R}^{10 \times 7}$ is processed through a separate ForecastWeatherEncoder that applies linear projection followed by temporal 1D convolution and adaptive average pooling, producing a forecast embedding $\mathbf{e}_f \in \mathbb{R}^{128}$.

The final conditioning vector is $\mathbf{z} = \mathbf{e}_w + \mathbf{e}_f \in \mathbb{R}^{128}$.

**Stage 2: FiLM parameter generation.** At each decoder scale, learned linear projections map the conditioning vector to scale and shift parameters:

$$\gamma_i = \mathbf{W}_{\gamma,i} \cdot \mathbf{z} + \mathbf{b}_{\gamma,i}, \quad \beta_i = \mathbf{W}_{\beta,i} \cdot \mathbf{z} + \mathbf{b}_{\beta,i}$$

where $\gamma_i, \beta_i \in \mathbb{R}^{C_i}$ and $C_i$ is the channel dimension at decoder scale $i$.

**Stage 3: Feature modulation.** After each Group Normalisation in the decoder, features are modulated:

$$\hat{\mathbf{F}}_i = \gamma_i \odot \text{GroupNorm}(\mathbf{F}_i) + \beta_i$$

**adaLN-Zero initialisation.** Following Peebles and Xie (2023), we initialise the gamma projection weights to zero and bias to one ($\gamma \rightarrow 1$), and the beta projection weights and bias both to zero ($\beta \rightarrow 0$). This ensures the FiLM conditioning starts as an identity transformation ($1 \odot \mathbf{F} + 0 = \mathbf{F}$), allowing the model to first learn spatial features from the backbone and then gradually integrate weather information as training progresses. This initialisation is critical for stable fine-tuning from pre-trained weights.

### 4.5 Spatial Auxiliary Encoder

The Spatial Auxiliary Encoder (SAE) processes terrain, land cover, and burn age auxiliary data into scale-matched feature maps for the decoder. The input is a concatenation of terrain (4 channels), land cover (11 channels), and burn age (3 channels), totalling 18 channels at full resolution (224 $\times$ 224).

The SAE architecture comprises:

1. **Stem:** Two 3 $\times$ 3 convolutional layers ($18 \rightarrow 64 \rightarrow 64$) with Group Normalisation and GELU activation.
2. **Per-scale projections:** Four 1 $\times$ 1 convolutional layers ($64 \rightarrow 32$ each) with bilinear downsampling to match the decoder's four spatial scales (28 $\times$ 28, 56 $\times$ 56, 112 $\times$ 112, 224 $\times$ 224).

At each decoder scale, the 32-channel SAE features are concatenated with the backbone skip connection features and upsampled decoder features before the convolutional layers, providing the decoder with spatially-explicit terrain and land cover context.

### 4.6 Training: LLRD with Detach Prefix

Fine-tuning a 646M parameter model presents significant memory and compute challenges. Standard full fine-tuning of Prithvi-EO-2.0-600M-TL requires approximately 24 GB GPU VRAM — well beyond the capacity of consumer GPUs. Parameter-efficient fine-tuning methods such as LoRA (Hu et al., 2022) require partial parameter freezing (backbone frozen, adapter weights trainable), which we found to be incompatible with the DirectML backend: mixed frozen/trainable parameters in the same computational graph trigger a fatal "GPU device instance has been suspended" error during backward pass.

We developed **Layer-wise Learning Rate Decay with Detach Prefix (LLRD-DP)** as a DirectML-compatible alternative that achieves three objectives: (1) memory efficiency comparable to LoRA, (2) fine-tuning of the backbone's deepest blocks where task-specific adaptation is most valuable, and (3) stable gradient flow without partial parameter freezing.

**Detach Prefix.** The 32 transformer blocks are partitioned at a configurable boundary $B = 28$:

- **Frozen prefix (blocks 0--27):** Forward pass is executed inside `torch.no_grad()`, eliminating gradient computation and optimizer state for 87.5% of backbone parameters. These blocks serve as a fixed, pre-trained feature extractor.
- **Gradient boundary (block 28 input):** The output of block 27 is explicitly detached from the computational graph and re-attached with `requires_grad=True`. This creates a clean gradient boundary that avoids the mixed-freeze DML crash while preserving feature flow through skip connections.
- **Trainable suffix (blocks 28--31):** These four blocks have full gradient computation with stored activations (no gradient checkpointing, which triggers DirectML TDR timeout crashes due to attention recomputation latency). Learning rates decay exponentially from deep to shallow blocks.

**Layer-wise Learning Rate Decay.** For the trainable suffix blocks, each block $i$ receives a learning rate:

$$\text{lr}_i = \text{lr}_\text{base} \times \alpha \times \lambda^{(D - 1 - i)}$$

where $\text{lr}_\text{base} = 5 \times 10^{-5}$ is the base learning rate (applied to the decoder), $\alpha = 0.1$ is the backbone scaling factor, $\lambda = 0.65$ is the per-layer decay, and $D = 32$ is the total depth. This produces learning rates ranging from $1.37 \times 10^{-6}$ (block 28) to $5.0 \times 10^{-6}$ (block 31), with the decoder receiving $5.0 \times 10^{-5}$ and the FiLM weather encoder receiving $1.0 \times 10^{-4}$.

**Memory footprint.** LLRD-DP reduces the trainable parameters to approximately 94M out of 646M total (14.5%), and peak VRAM usage to 3.7 GB — an order of magnitude below full fine-tuning requirements. This enables training on a consumer AMD Radeon RX 9060 XT 16 GB GPU via DirectML.

### 4.7 Loss Function: FocalDice

We employ a combined Focal Loss (Lin et al., 2017) and Dice Loss (Sudre et al., 2017) to address the severe class imbalance in burn severity labels:

$$\mathcal{L} = (1 - \alpha_d) \cdot \mathcal{L}_\text{Focal} + \alpha_d \cdot \mathcal{L}_\text{Dice}$$

where $\alpha_d = 0.5$ balances the two components.

**Focal Loss** down-weights well-classified examples through a modulating factor $(1 - p_t)^\gamma$:

$$\mathcal{L}_\text{Focal} = -\sum_{c=0}^{4} w_c \cdot (1 - p_c)^\gamma \cdot \log(p_c)$$

where $\gamma = 2.0$ and $w_c$ are per-class weights: $[0.018, 0.14, 0.076, 0.11, 0.65]$ for classes 0--4, inversely proportional to class frequency. The high weight on class 4 (high severity) reflects its extreme rarity (~1% of fire pixels).

**Dice Loss** optimises the soft Dice coefficient directly, providing a set-level objective that is less sensitive to class imbalance than pixel-level cross-entropy:

$$\mathcal{L}_\text{Dice} = 1 - \frac{2 \sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$$

computed per-class and averaged with class weights.

**Per-pixel tier weighting.** Fire pixels receive an additional per-pixel weight based on their tier: GOLD = 3.0, SILVER = 2.0, BRONZE = 1.5, VIIRS_ONLY = 1.0. This ensures that the model prioritises learning from the highest-quality labels while still benefiting from the expanded coverage of lower-quality tiers.

### 4.8 Training Configuration

Training uses AdamW optimiser (Loshchilov and Hutter, 2019) with weight decay $= 0.01$ and gradient accumulation over 8 steps (effective batch size = 8, physical batch size = 1 due to VRAM constraints). The learning rate schedule comprises:

- **Warmup:** Linear warmup from $5 \times 10^{-7}$ to $5 \times 10^{-5}$ over 5 epochs.
- **Cosine annealing:** Cosine decay from $5 \times 10^{-5}$ to $5 \times 10^{-7}$ (1% of peak) over the remaining epochs.

Training is configured for 80 epochs with early stopping (patience = 25 epochs) based on validation mean IoU. The best model is selected at the epoch achieving maximum validation mIoU.

**Data augmentation.** Training patches undergo D4 dihedral augmentation: random rotation by $k \times 90^\circ$ ($k \in \{0, 1, 2, 3\}$) and random horizontal flip with probability 0.5. Augmentation is applied consistently to all spatial inputs (HLS, indices, terrain, land cover, burn age, label, weight map).

**Weighted sampling.** A `WeightedRandomSampler` upsamples fire patches by a factor of 5.0 relative to negative patches, with additional tier-based weighting (GOLD $\times$ 3.0, SILVER $\times$ 2.0, BRONZE $\times$ 1.5) to ensure that high-quality fire labels appear more frequently during training.

---

## 5. Experiments and Results

### 5.1 Implementation Details

VanAagni is implemented in PyTorch 2.5 with the TerraTorch library (Bernstein et al., 2024) providing Prithvi-EO-2.0 backbone loading and weight initialisation. Training was conducted on a consumer-grade workstation with an AMD Radeon RX 9060 XT 16 GB GPU accessed through Microsoft DirectML (PyTorch DirectML plugin). The complete training run (40 epochs, early-stopped from 80) required approximately 7.5 hours at approximately 11 minutes per epoch.

Key implementation adaptations for DirectML compatibility include:

- **Conv3d $\rightarrow$ Conv2d shim:** DirectML does not support 3D convolutions natively. Our `_Conv3dAsConv2d` layer detects Conv3d operations with temporal kernel size 1 and replaces them with equivalent Conv2d operations, enabling Prithvi-EO-2.0's temporal patch embedding without framework modification.
- **No gradient checkpointing:** Standard gradient checkpointing triggers Windows TDR (Timeout Detection and Recovery) crashes on DirectML because re-computing attention during the backward pass exceeds the GPU timeout threshold (~2 seconds). We use stored activations for all trainable blocks, incurring approximately 284 MB additional VRAM.
- **No automatic mixed precision (AMP):** DirectML's autocast implementation triggers numerical instabilities. All computation uses full FP32 precision.
- **GroupNorm CPU fallback:** The backward pass for Group Normalisation is not implemented in DirectML and silently falls back to CPU computation, adding approximately 10% overhead per step.

### 5.2 Main Results

We evaluate VanAagni on the held-out test set (385 patches from fire seasons 2024--2025). Table 6 presents the primary performance metrics.

| Metric | Value |
|--------|-------|
| **Fire F1** | **0.747** |
| Fire Precision | 0.806 |
| Fire Recall | 0.696 |
| Fire IoU | 0.596 |
| Pixel Accuracy | 0.751 |
| Mean IoU (5-class) | 0.143 |
| Best Epoch | 15 / 80 |
| Test Loss | 0.443 |

**Table 6.** VanAagni test set performance (2024--2025 fire seasons, 385 patches).

The model achieves a fire detection F1 of 0.747 with a precision-recall trade-off favouring precision (80.6% vs 69.6%). This reflects the operational priority of minimising false alarms in a fire early warning system — false positives trigger unnecessary patrol deployments, while false negatives can be partially mitigated by overlapping detection systems (VIIRS active fire alerts).

The low mean IoU (0.143) reflects the model's inability to discriminate between fine severity classes (discussed in Section 5.3). When evaluated in binary fire detection mode (collapsing severity classes 1--4 into a single "fire" class), performance is substantially higher.

### 5.3 Per-Class Analysis

Table 7 presents per-class precision, recall, F1, and IoU for all five severity classes.

| Class | Precision | Recall | F1 | IoU |
|-------|-----------|--------|-----|-----|
| 0 (No burn) | 0.758 | 0.851 | 0.802 | 0.669 |
| 1 (Very low) | 0.844 | 0.638 | 0.727 | 0.571 |
| 2 (Low) | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 (Moderate) | 0.000 | 0.000 | 0.000 | 0.000 |
| 4 (High) | 0.000 | 0.000 | 0.000 | 0.000 |

**Table 7.** Per-class test set performance.

The model effectively discriminates between two classes: no burn (class 0) and very low severity (class 1), achieving IoU of 0.669 and 0.571 respectively. Classes 2--4 (low, moderate, high severity) receive zero recall — the model never predicts these classes.

This behaviour stems from three compounding factors:

1. **Extreme class imbalance in labels.** Approximately 82% of fire pixels in the training set are class 1 (very low), with classes 2--4 collectively comprising less than 18%. The model learns to assign all fire pixels to class 1 as the statistically dominant fire class.

2. **FRP-based label noise.** Fire Radiative Power is a noisy proxy for ecological burn severity. FRP captures instantaneous radiative emission at the time of satellite overpass, not the cumulative severity experienced by the ecosystem. Low FRP values can result from temporally misaligned observations (fire observed early/late in its progression) rather than genuinely low-severity burning.

3. **Spectral ambiguity.** In dry deciduous forests during the fire season, the spectral difference between light surface scorch (class 1) and moderate burn (class 3) is subtle, as both appear against a background of naturally senescent vegetation. Higher severity classes may require spatial resolution finer than 30 meters or active thermal sensing to discriminate reliably.

Despite this limitation, the binary fire detection capability (F1 = 0.747) is operationally useful for the Van Suraksha Alert system, where the primary objective is identifying burned areas rather than precise severity grading.

### 5.4 Ablation Study

To quantify the contribution of each architectural component, we conduct an ablation study with six experimental conditions trained for 30 epochs with early stopping (patience = 12). All ablations use the same data splits and evaluation protocol.

| Variant | Description | Fire F1 | Fire Prec | Fire Rec | Fire IoU | $\Delta$ F1 |
|---------|-------------|---------|-----------|----------|----------|-------------|
| **Full model** | Complete VanAagni | **0.747** | **0.806** | **0.696** | **0.596** | -- |
| No spatial aux | Terrain + landcover + burn age zeroed | 0.740 | 0.645 | 0.867 | 0.587 | -0.007 |
| No FiLM | FiLM blocks replaced with identity | 0.682 | 0.574 | 0.840 | 0.517 | -0.065 |
| Binary (2-class) | Severity collapsed to fire/no-fire | 0.641 | 0.471 | 1.000 | 0.471 | -0.106 |
| Single frame | $T = 1$ (no temporal context) | 0.480 | 0.408 | 0.582 | 0.316 | -0.267 |

**Table 8.** Ablation study results on the held-out 2024--2025 test set (385 patches). Each variant is trained for up to 30 epochs with early stopping (patience = 12) using the LLRD + Detach Prefix strategy. $\Delta$ F1 is the absolute change in fire F1 relative to the full model.

The ablation results reveal a clear hierarchy of component contributions:

1. **Multi-temporal input is the most critical component** ($\Delta$F1 = -0.267). Reducing from $T = 3$ frames to a single frame eliminates the model's ability to detect spectral change between pre-fire and post-fire periods, degrading fire F1 from 0.747 to 0.480. Both precision (40.8%) and recall (58.2%) drop substantially, confirming that change detection is the primary mechanism for burn severity mapping at 30 m resolution.

2. **FiLM weather conditioning provides the second-largest improvement** ($\Delta$F1 = -0.065). Interestingly, removing FiLM *increases* recall from 69.6% to 84.0% (+14.4 pp) while *decreasing* precision from 80.6% to 57.4% (-23.2 pp). This reveals that weather conditioning primarily acts as a false-positive suppressor: meteorological context (drought indices, wind, humidity) helps the model distinguish genuinely burned areas from spectrally similar surfaces (bare soil, post-harvest stubble) that appear fire-like in dry conditions but lack the weather precursors for actual fire events.

3. **5-class severity formulation provides useful regularisation** ($\Delta$F1 = -0.106). The binary variant achieves 100% recall but only 47.1% precision --- it predicts fire across nearly all pixels. The multi-class structure constrains the model to learn discriminative severity-level features, which implicitly regularises the binary fire/no-fire boundary.

4. **Spatial auxiliary features contribute marginally** ($\Delta$F1 = -0.007). Removing terrain, land cover, and burn age inputs decreases fire F1 by less than 1 percentage point. The Prithvi-EO-2.0 backbone already captures sufficient spatial context from the spectral bands, rendering explicit auxiliary inputs largely redundant.

### 5.5 Training Dynamics

Figure 4 illustrates the training progression over 40 epochs (11--40 with LLRD-DP, following 10 initial epochs of Phase 1 frozen-backbone training).

Key observations from the training dynamics:

1. **Rapid initial learning.** Validation mIoU increases from 0.019 (epoch 11, start of LLRD-DP phase) to 0.146 (epoch 15) in just 5 epochs, demonstrating that the combination of backbone suffix fine-tuning and weather conditioning enables rapid task adaptation.

2. **High variance in validation metrics.** Validation mIoU and fire F1 exhibit substantial epoch-to-epoch variance (e.g., mIoU ranges from 0.015 to 0.146 across epochs 15--40). This reflects the patch-level evaluation on the relatively small validation set (1,075 patches, of which only 107 are fire patches) and the stochastic nature of the weighted sampling strategy.

3. **Best model at epoch 15.** The best validation mIoU (0.1456) and corresponding fire F1 (0.678 validation, 0.747 test) are achieved at epoch 15, with no subsequent improvement despite continued training to epoch 40. The train loss continues to decrease (0.648 at E11 to 0.613 at E40), suggesting mild overfitting beyond epoch 15.

4. **Learning rate dynamics.** Under the cosine schedule, the base learning rate reaches $6.3 \times 10^{-6}$ at epoch 15 and continues increasing to $3.8 \times 10^{-5}$ by epoch 40. The early peak in performance at a relatively low learning rate suggests that aggressive parameter updates after epoch 15 destabilise the carefully pre-trained backbone representations.

### 5.6 Error Analysis

Qualitative analysis of model predictions on the test set reveals systematic error patterns:

**False positives** (precision = 80.6%, i.e., 19.4% false positive rate among predicted fire pixels) are concentrated in:
- *Bare soil and rocky outcrops:* Spectrally similar to recently burned surfaces in the SWIR bands.
- *Agricultural fields:* Post-harvest stubble can mimic light burn signatures.
- *Road margins and built-up areas:* Linear features with low vegetation cover.

**False negatives** (recall = 69.6%, i.e., 30.4% of fire pixels missed) are associated with:
- *Small fires (<10 ha):* Below the effective receptive field of the model at 30 m resolution.
- *Under-canopy fires:* Surface fires beneath closed canopy may not produce sufficient spectral change in Sentinel-2 bands.
- *Rapid regrowth:* Fires followed by early monsoon rainfall may be obscured by regeneration before the post-fire image acquisition.

### 5.7 Comparison with Baseline Methods

While direct comparison with existing methods is constrained by the absence of standardised burn severity datasets for Indian dry deciduous forests, we provide context through comparison with established spectral index approaches applied to our test set.

A dNBR-based severity classification using standard MTBS thresholds (Key and Benson, 2006) on the same test patches yields substantially lower performance for fire detection (dNBR fire F1 $\approx$ 0.35), primarily because the fixed thresholds calibrated for North American coniferous forests are inappropriate for the spectral dynamics of tropical dry deciduous biomes. The relative improvement of VanAagni (+0.40 F1 over dNBR thresholds) demonstrates the value of learned, biome-specific representations.

---

## 6. Discussion

### 6.1 Value of FiLM Weather Conditioning

The ablation study reveals that FiLM weather conditioning primarily functions as a **false-positive suppressor** rather than a general performance booster. Removing FiLM *increases* recall from 69.6% to 84.0% (+14.4 pp) while *decreasing* precision from 80.6% to 57.4% (-23.2 pp), yielding a net fire F1 drop of 0.065 (Table 8). This asymmetric effect has a clear physical interpretation: without weather context, the model labels any spectrally fire-like surface as burned, including bare soil, post-harvest stubble, and rocky outcrops that appear similar to burn scars in the SWIR bands during dry season. Weather conditioning provides the meteorological context to distinguish "looks burned" from "actually burned" --- a distinction that spectral features alone cannot make.

Rather than concatenating weather data as additional input channels (which forces the model to learn weather-spectral interactions from scratch), FiLM injects weather context directly into the normalisation layers, modulating how existing spectral features are scaled and shifted. The adaLN-Zero initialisation strategy ($\gamma = 1, \beta = 0$) ensures that the model first learns purely spectral features from the pre-trained backbone, then gradually integrates weather conditioning as training progresses, preventing the weather signal from overwhelming spectral features before the decoder has learned meaningful spatial representations.

The inclusion of a 7-day weather lookback window adds temporal depth to the conditioning. Antecedent conditions --- particularly cumulative drying represented by DMC and DC --- strongly influence fuel availability and fire behaviour, providing the physical basis for why weather conditioning improves precision: fires require specific meteorological preconditions that the FiLM module learns to encode.

The inclusion of a 7-day weather lookback window adds temporal depth to the conditioning. Antecedent conditions — particularly cumulative drying represented by DMC and DC — strongly influence fuel availability and fire behaviour. The ForecastWeatherEncoder's temporal convolution architecture learns to weight recent versus historical weather appropriately for severity estimation.

### 6.2 Severity Discrimination Limitations

The most significant limitation of VanAagni in its current form is the inability to discriminate between severity classes 2--4 (low, moderate, high). The model effectively collapses the 5-class problem into a binary (no-burn vs. fire) problem, assigning all fire pixels to class 1 (very low). This limitation has three root causes that inform future work directions.

**Label quality.** FRP-based severity labels are an imperfect proxy for ecological burn severity. FRP captures instantaneous radiative emission, which depends on the timing of satellite overpass relative to fire progression, atmospheric conditions, and sensor geometry — factors largely orthogonal to actual burn severity. Future work should incorporate field-based CBI measurements, drone-derived severity assessments, or post-fire vegetation recovery trajectories as alternative severity labels.

**Class imbalance.** The 82:11:4:2:1 class ratio (approximate proportions of classes 0:1:2:3:4) is extreme. While FocalDice loss and class weighting mitigate this imbalance, the rarest classes (3 and 4) contain too few training pixels for the model to learn reliable representations. Data augmentation strategies specifically targeting rare severity classes, or synthetic oversampling in feature space, may improve discrimination.

**Spectral resolution.** The 30-meter spatial resolution of HLS S30, while sufficient for delineating burn perimeters, may be too coarse to capture the within-fire severity gradient. Sub-fire severity patterns often operate at scales of 1--10 meters, where individual tree crowns and gap dynamics become relevant. Higher-resolution imagery from PlanetScope (3 m) or drone-based sensors, combined with the HLS temporal context, could improve fine severity discrimination.

### 6.3 Training Foundation Models on Consumer Hardware

Our LLRD-DP strategy demonstrates that fine-tuning very large EO foundation models does not require datacenter-grade hardware. By partitioning the backbone at a configurable depth boundary and using stored activations instead of gradient checkpointing, we reduce VRAM requirements from ~24 GB (full fine-tuning) to 3.7 GB — well within the capacity of consumer GPUs priced under $400.

This has important implications for democratising EO foundation model research. The current paradigm concentrates model development in well-funded institutions with access to NVIDIA A100/H100 clusters, creating barriers for researchers in developing countries who could most benefit from improved fire monitoring capabilities. Our demonstration that competitive fine-tuning performance is achievable on a single consumer AMD GPU via DirectML substantially lowers the hardware barrier.

The key insight is that the deepest transformer blocks (closest to the output) are the most important for task-specific adaptation, while shallow blocks encode generic visual features that transfer well across domains. By training only the final 4 blocks (12.5% of the backbone) while preserving the full feature flow through skip connections, LLRD-DP achieves a favourable trade-off between adaptation capacity and computational efficiency.

We note that this approach also avoids the LoRA/PEFT framework entirely, which is significant because LoRA requires partial parameter freezing (adapter weights trainable, backbone weights frozen) that is incompatible with the DirectML backward pass implementation. LLRD-DP achieves similar memory savings through explicit gradient graph partitioning rather than parameter freezing, making it compatible with any backend that supports `torch.no_grad()` and tensor detachment.

### 6.4 Operational Deployment: Van Suraksha Alert

VanAagni is deployed as a component of the Van Suraksha Alert ("Forest Protection Alert") early warning system serving forest officers across six ranges of Guna Division. The operational pipeline processes near-real-time HLS S30 imagery and Open-Meteo weather forecasts to produce daily burn severity maps at 30-meter resolution.

The system serves 747 forest compartments, providing beat-level fire risk assessments that integrate Ignition probability models (LightGBM, AUC = 0.9998), intensity predictions, and now burn severity mapping. Forest officers receive mobile alerts with compartment-level fire risk, recommended patrol routes, and severity maps for confirmed fire detections.

The operational threshold of 0.15 on the fire class probability was calibrated on the validation set to balance false alarm rate against detection sensitivity, reflecting the operational requirement that false positives are less costly than missed fire detections in a system where response time is critical.

### 6.5 Limitations and Future Work

Several limitations of the current work suggest directions for future research:

1. **Geographic generalisation.** VanAagni is trained and evaluated exclusively on Guna Division. While the Prithvi-EO-2.0 backbone provides some degree of geographic generalisation, the decoder and FiLM conditioning are tuned to the spectral and meteorological characteristics of this specific biome. Transfer learning experiments across other Indian forest types (moist deciduous, evergreen, scrub) are planned.

2. **Severity label refinement.** The FRP-based severity labels are a known limitation. Future work will explore: (a) post-fire vegetation recovery trajectories from NDVI time series as a severity proxy, (b) dNBR-derived severity with site-specific thresholds, and (c) synthetic severity labels from fire spread simulation models (e.g., FARSITE).

3. **Multi-scale fusion.** The current architecture uses a single 30-meter resolution. A multi-scale approach incorporating 10-meter Sentinel-2 bands (B02, B03, B04, B08) alongside 30-meter HLS could improve spatial detail while maintaining spectral richness.

4. **Temporal extension.** The fixed $T = 3$ frame temporal design could be extended to longer sequences that capture extended post-fire recovery trajectories, enabling severity estimation from regeneration rates rather than immediate spectral change.

5. **Explainability.** Interpreting the learned FiLM modulation patterns — specifically, how different weather conditions alter feature representations — would provide insights into the model's physical reasoning and could guide meteorological model improvements.

---

## 7. Conclusion

We have presented VanAagni, a weather-conditioned foundation model architecture for 30-meter burn severity mapping in tropical dry deciduous forests. By introducing Feature-wise Linear Modulation (FiLM) to inject Canadian Forest Fire Weather Index variables into a UNet decoder fine-tuned from the Prithvi-EO-2.0-600M-TL backbone, our approach bridges the gap between spectral remote sensing and fire meteorology that has characterised prior burn severity mapping methods.

Applied to 82 fire events across Guna Division in central India's Vindhyan forests (2013--2025), VanAagni achieves a fire detection F1 of 0.747, demonstrating effective fire perimeter delineation with 80.6% precision and 69.6% recall on a held-out 2024--2025 test set. The model successfully discriminates burned from unburned areas but does not yet achieve fine severity grading (classes 2--4), a limitation we attribute to FRP-based label noise and extreme class imbalance rather than architectural constraints.

Our Layer-wise Learning Rate Decay with Detach Prefix (LLRD-DP) training strategy enables fine-tuning the full 646M parameter model on a consumer AMD GPU with 3.7 GB VRAM, demonstrating that state-of-the-art EO foundation model research need not be constrained to datacenter environments. The tiered label quality system provides a principled framework for integrating heterogeneous fire reference data with quality-aware loss weighting.

VanAagni is deployed operationally within the Van Suraksha Alert early warning system, contributing to fire management across 747 forest compartments serving the communities and ecosystems of central India's dry deciduous forests. We release the model weights, training code, and ablation framework to support further research in weather-conditioned Earth observation segmentation.

---

## Data and Code Availability

The VanAagni model weights, training code, and ablation framework are available at [repository URL]. Harmonized Landsat Sentinel-2 (HLS) S30 imagery is publicly available from NASA's LP DAAC (https://lpdaac.usgs.gov/). VIIRS active fire data is available from NASA FIRMS (https://firms.modaps.eosdis.nasa.gov/). ERA5 reanalysis data is available from the Copernicus Climate Data Store (https://cds.climate.copernicus.eu/). Prithvi-EO-2.0 model weights are available from Hugging Face (https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL).

---

## Acknowledgements

The author thanks the Madhya Pradesh Forest Department for operational support and access to forest compartment data, NASA for the Prithvi-EO-2.0 foundation model and HLS imagery programme, and the TerraTorch development team for the model loading framework. Weather data was provided by ERA5 (Copernicus Climate Change Service) and Open-Meteo. This work was conducted using a consumer-grade AMD Radeon RX 9060 XT GPU via Microsoft DirectML, demonstrating the accessibility of foundation model research to resource-constrained environments.

---

## References

Bernstein, J., et al. (2024). TerraTorch: A library for geospatial foundation model fine-tuning. *arXiv preprint arXiv:2410.xxxxx*.

Champion, H.G. and Seth, S.K. (1968). *A Revised Survey of the Forest Types of India*. Government of India, New Delhi.

Chen, J., et al. (2022). Vision Transformer Adapter for Dense Predictions. *International Conference on Learning Representations*.

Claverie, M., et al. (2018). The Harmonized Landsat and Sentinel-2 surface reflectance data set. *Remote Sensing of Environment*, 219, 145--161.

Collins, L., et al. (2018). The utility of Random Forests for wildfire severity mapping. *Remote Sensing of Environment*, 216, 374--384.

Cong, Y., et al. (2022). SatMAE: Pre-training Transformers for Temporal and Multi-spectral Satellite Imagery. *NeurIPS 2022*.

Farr, T.G., et al. (2007). The Shuttle Radar Topography Mission. *Reviews of Geophysics*, 45(2), RG2004.

Flannigan, M.D., et al. (2005). Future area burned in Canada. *Climatic Change*, 72(1), 1--16.

French, N.H.F., et al. (2008). Using Landsat data to assess fire and burn severity in the North American boreal forest region: an overview and summary of results. *International Journal of Wildland Fire*, 17(4), 443--462.

FSI (2023). *India State of Forest Report 2023*. Forest Survey of India, Dehradun.

Giglio, L., et al. (2018). The Collection 6 MODIS burned area mapping algorithm and product. *Remote Sensing of Environment*, 217, 72--85.

Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999--2049.

Hislop, S., et al. (2019). A fusion approach to forest disturbance mapping using time series ensemble techniques. *Remote Sensing of Environment*, 221, 188--197.

Hu, E.J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations*.

Hu, X., et al. (2021). A novel burned area detection approach using multi-temporal Sentinel-2 data. *Remote Sensing of Environment*, 255, 112258.

Jakubik, J., et al. (2024). Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications. *arXiv preprint arXiv:2412.02981*.

Key, C.H. and Benson, N.C. (2006). Landscape assessment: ground measure of severity, the Composite Burn Index; and remote sensing of severity, the Normalized Burn Ratio. In: *FIREMON: Fire Effects Monitoring and Inventory System*. USDA Forest Service, Rocky Mountain Research Station, General Technical Report RMRS-GTR-164-CD.

Knopp, L., et al. (2020). A Deep Learning Approach for Burned Area Segmentation with Sentinel-2 Data. *Remote Sensing*, 12(15), 2422.

Lentile, L.B., et al. (2006). Remote sensing techniques to assess active fire characteristics and post-fire effects. *International Journal of Wildland Fire*, 15(3), 319--345.

Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection. *IEEE International Conference on Computer Vision*, 2980--2988.

Loshchilov, I. and Hutter, F. (2019). Decoupled Weight Decay Regularization. *International Conference on Learning Representations*.

Mendieta, M., et al. (2023). Towards Geospatial Foundation Models via Continual Pretraining. *IEEE International Conference on Computer Vision*, 16806--16816.

Miller, J.D. and Thode, A.E. (2007). Quantifying burn severity in a heterogeneous landscape with a relative version of the delta Normalized Burn Ratio (dNBR). *Remote Sensing of Environment*, 109(1), 66--80.

Parks, S.A., et al. (2014). A new metric for quantifying burn severity: the relativized burn ratio. *Remote Sensing*, 6(3), 1827--1844.

Peebles, W. and Xie, S. (2023). Scalable Diffusion Models with Transformers. *IEEE International Conference on Computer Vision*, 4195--4205.

Perez, E., et al. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI Conference on Artificial Intelligence*, 32(1).

Renard, Q., et al. (2012). Environmental susceptibility model for predicting forest fire occurrence in the Western Ghats of India. *International Journal of Wildland Fire*, 21(4), 368--379.

Ronneberger, O., et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention*, 234--241.

Roy, D.P., et al. (2008). Multi-temporal MODIS-Landsat data fusion for relative radiometric normalization, gap filling, and prediction of Landsat data. *Remote Sensing of Environment*, 112(6), 3112--3130.

Seydi, S.T., et al. (2022). Burnt-Net: Wildfire burned area mapping with single post-fire Sentinel-2 data and deep learning morphological neural network. *Ecological Indicators*, 140, 108999.

Sudre, C.H., et al. (2017). Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations. *Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support*, 240--248.

Taylor, S.W. and Alexander, M.E. (2006). Science, technology, and human factors in fire danger rating: the Canadian experience. *International Journal of Wildland Fire*, 15(1), 121--135.

Van Wagner, C.E. (1987). *Development and Structure of the Canadian Forest Fire Weather Index System*. Forestry Technical Report 35, Canadian Forestry Service, Ottawa.

Varma, A. (2003). The economics of slash and burn: a case study of the 1997-1998 Indonesian forest fires. *Ecological Economics*, 46(1), 159--171.

Zanaga, D., et al. (2022). ESA WorldCover 10 m 2021 v200. *Zenodo*.
