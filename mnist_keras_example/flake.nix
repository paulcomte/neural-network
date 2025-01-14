{
  description = "A very basic flake";

  inputs = {
    nixpkgs-stable.url = "github:nixos/nixpkgs/nixos-24.11";
  };

  outputs = { self, nixpkgs-stable }: {
    devShell = {
      # Define a development shell for aarch64-darwin
      aarch64-darwin = let
        pkgs = import nixpkgs-stable {
          system = "aarch64-darwin";
        };

      in pkgs.mkShell {
        venvDir = ".venv";
        packages = with pkgs; [
          python3
          pypy
        ] ++ (with pkgs.python3Packages; [
          pip
          python-lsp-server
          venvShellHook
        ]);

        shellHook = ''
            if [ ! -d "$venvDir" ]; then
              python3 -m venv $venvDir
            fi
            source $venvDir/bin/activate
            pip install -r requirements.txt || echo "requirements.txt not found or pip install failed"

            ssh_fix="import ssl\nssl._create_default_https_context = ssl._create_unverified_context"
            ssh_fix_file=".venv/lib/python3.12/site-packages/keras/src/utils/file_utils.py"

            if ! grep -qxF "import ssl" "$ssh_fix_file"; then
              echo "$ssh_fix" | sed -i "1i$(cat)" "$ssh_fix_file"
            fi
          '';
      };
    };
  };
}
