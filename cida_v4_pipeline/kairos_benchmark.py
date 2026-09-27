from kairos_benchmark import *

if __name__ == "__main__":
    import sys
    # Delegate directly to the main script
    from kairos_benchmark import __doc__ as script_doc
    # Run the exact same CLI
    import argparse
    parser = argparse.ArgumentParser(description=script_doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["download", "cache", "train", "report", "smoke_test"])
    parser.add_argument("--track", choices=list(TRACKS.keys()), help="Target benchmark track ('text' or 'vision')")
    parser.add_argument("--variant", choices=["linear_probe", "equal_capacity", "kairos"], help="Model variant to train")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--d_in", type=int, default=None, help="Input dimension (auto-detected from shard if omitted)")
    parser.add_argument("--out_dir", type=str, default="./kairos_checkpoints", help="Output directory for results")
    args = parser.parse_args()

    if args.stage == "download":
        for name, t in TRACKS.items():
            print(f"[download] Starting track {name}: {t.backbone_repo}")
            download_backbone(t)
    elif args.stage == "cache":
        if not args.track:
            raise SystemExit("--track is required (choices: 'text', 'vision')")
        t = TRACKS[args.track]
        bb = download_backbone(t)
        cache_latents(t, bb, t.train_split, f"./latent_cache/{t.name}/train")
        cache_latents(t, bb, t.eval_split, f"./latent_cache/{t.name}/eval")
    elif args.stage == "train":
        if not (args.track and args.variant):
            raise SystemExit("--track and --variant are required for stage 'train'")
        t = TRACKS[args.track]
        train_variant(
            t,
            args.variant,
            shard_dir=f"./latent_cache/{t.name}/train",
            eval_shard_dir=f"./latent_cache/{t.name}/eval",
            d_in=args.d_in,
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    elif args.stage == "report":
        report(out_dir=args.out_dir)
    elif args.stage == "smoke_test":
        run_smoke_test()
