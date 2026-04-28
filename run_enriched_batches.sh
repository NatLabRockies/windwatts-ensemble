#!/bin/bash
# Run enriched experiment in batches of 24 stations.
# Usage: bash run_enriched_batches.sh [start_batch]
# Start from batch N (default: 1). Batches already completed are skipped.

set -euo pipefail
cd "$(dirname "$0")"

INFILE="data/e2e/training/combined_quantiles_long_with_topo_loocv_10km.csv"
GWA="data/e2e/training/site_height_ws_avg_with_gwa.csv"
OUTDIR="data/output/enriched_batches"
mkdir -p "$OUTDIR"

# All 12 batches
BATCHES=(
  "10,11,12,128,13,130,131,132,134,135,136,137,138,140,143,144,145,147,148,150,151,152,153,154"
  "157,159,160,161,163,165,167,169,17,170,171,172,173,176,177,179,180,181,182,183,184,185,187,188"
  "194,195,197,199,234,237,238,241,261,262,263,264,28,3,37,38,388,39,40,418,42,422,43,430"
  "44,47,48,5,6,62,63,64,65,66,67,68,69,7,70,71,72,73,74,75,77,79,8,81"
  "82,83,84,85,86,87,88,89,9,90,91,92,al_cedarpoint,al_fortmorgan,al_middlebay,al_perdidopass,ca_angelsgate,ca_lajolla,ca_losangelesbadger,ca_losangelesberth161,ca_losangelespier400,ca_losangelespierf,ca_losangelespierj,ca_martinez"
  "ca_northjettylanding,ca_northspit,ca_portsanluis,ca_sanfrancisco,ca_santabarbara,ca_santamonica,ca_tonzi,co_niwotridge,co_nwtc,ct_newlondon,de_brandywineshoal,de_lewes,findlayv1,findlayv2,findlayw1,findlayw2,findlayz1,findlayz2,findlayz3,fl_aripeka,fl_bigcarlospass,fl_cedarkey,fl_clambayou,fl_eastbay"
  "fl_eastpoint,fl_foweyrock,fl_keatonbeach,fl_keywest,fl_molassesreef,fl_stjohnsriver,fl_tampa,fl_venice,greenvillewg1,greenvillewg2,greenvillewg3,harpsterwtg1,il_argonne,il_calumetharbor,il_northerlyisle,il_pioneertrail,il_pioneertrail2,in_burnsharbor,in_michigancity,ks_meridianway,ks_meridianway2,ks_meridianway4,ks_meridianway6,ks_meridianway7"
  "la_ameradapass,la_bayougauche,la_berwick,la_calcasieupass,la_frenierlanding,la_freshwatercanal,la_marshisland,la_newcanalstation,la_pilotsstation,la_pilottown,la_shellbeach,la_southtimbalierblock,la_southwestpass,la_terrebonnebay,ma_blandford,ma_bordenflats,ma_buzzardsbay,marionw1,marionw2,marionw3,me_mountdesertrock,mi_bigbay,mi_detourvillage,mi_fairport"
  "mi_fortgratiot,mi_grandtraverse,mi_grandtraversebay,mi_gravellyshoal,mi_holland,mi_isleroyale,mi_littlerapids,mi_mackinawcity,mi_manisteeharbor,mi_menominee,mi_muskegon,mi_naubinway,mi_passageisland,mi_portsanilac,mi_saginawbay,mi_spectaclereef,mi_stclairshores,mi_tawaspoint,mi_thunderbayisland,mi_whitefishpoint,mi_whiteshoal,mn_duluth,mn_grandportage,mn_prairiestar"
  "mn_silverbay,mo_ozark,ms_baywaveland,nc_capelookout,nd_dickinson,nj_robbinsreef,ny_barcelona,ny_brookhaven,ny_dunkirk,ny_marinersharbor,ny_niagara,ny_olcottharbor,ny_oswego,ny_rochester,oh_conneaut,oh_fairport,oh_huronharbor,oh_marblehead,oh_southbassisland,oh_toledo,oh_toledocrib,oh_toledolight,ok_arbucklemountain,ok_arbucklemountain2"
  "ok_arbucklemountain3,ok_arbucklemountain4,ok_balko,ok_sgp,or_butlergrade,or_sevenmile,or_tillamook,or_troutdale,or_wasco,ottawaw1,pauldingl2,pauldingl3,ri_providence,sc_follyisland,sc_spiderweb,tx_baffinbay,tx_champion,tx_champion5,tx_champion6,tx_freeport,tx_galvestonbay,tx_galvestonbridge,tx_matagordabay,tx_moda"
  "tx_packerychannel,tx_panhandle,tx_portaransas,tx_rincondelsanjose,tx_rolloverpass,tx_sabinepass,tx_sarita,va_capehenry,va_rappahannock,va_yorktown,vt_proctormaple,wa_destructionisland,wa_hanford,wa_naselleridge,wi_deathsdoor,wi_devilsisland,wi_kenosha,wi_portwashington,wi_portwing,wi_saxonharbor,wi_sheboygan"
)

START=${1:-1}

for i in "${!BATCHES[@]}"; do
    BATCH_NUM=$((i + 1))
    if [ "$BATCH_NUM" -lt "$START" ]; then
        continue
    fi

    OUTFILE="$OUTDIR/batch$(printf '%02d' $BATCH_NUM).csv"
    if [ -f "$OUTFILE" ]; then
        echo "[$(date +%H:%M:%S)] Batch $BATCH_NUM already exists, skipping: $OUTFILE"
        continue
    fi

    echo "[$(date +%H:%M:%S)] Starting batch $BATCH_NUM / ${#BATCHES[@]} ..."
    python -m wem.experiment.runner enriched \
        --infile "$INFILE" \
        --outfile "$OUTFILE" \
        --gwa-file "$GWA" \
        --include-gwa \
        --n-jobs 4 \
        --xgb-threads 3 \
        --stations "${BATCHES[$i]}" \
        --overwrite

    echo "[$(date +%H:%M:%S)] Batch $BATCH_NUM complete: $OUTFILE"

    # Cumulative comparison against baseline
    echo ""
    echo "[$(date +%H:%M:%S)] === Cumulative comparison after batch $BATCH_NUM ==="
    python -m wem.experiment.compare \
        --baseline data/reference/loocv/ml_results.csv \
        --experiments "$OUTDIR"/batch*.csv \
        --labels Enriched \
        --top-n 5
    echo ""
done

echo "[$(date +%H:%M:%S)] All batches done."
