{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: {
    devShell = {
      # Define a development shell for aarch64-darwin
      aarch64-darwin = let
        pkgs = import nixpkgs {
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
          '';
      };
    };
  };
}
