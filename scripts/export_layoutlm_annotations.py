from app.annotation_export import export_annotation_folder

if __name__ == "__main__":
    exported, manifest = export_annotation_folder()
    print(f"Exported {len(exported)} files")
    print(f"Manifest: {manifest}")