{
  description = "Formally verified Transformer architecture in Rocq";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        rocq = pkgs.rocqPackages_9_1.rocq-core;
        stdlib = pkgs.rocqPackages_9_1.stdlib;

      in {
        packages = {
          default = pkgs.stdenv.mkDerivation {
            pname = "rocq-transformer";
            version = "0.1.0";

            src = ./.;

            nativeBuildInputs = [ rocq stdlib ];

            buildPhase = ''
              make all
            '';

            installPhase = ''
              mkdir -p $out/lib/rocq/${rocq.version}/user-contrib/Transformer
              cp -r Transformer/*.vo $out/lib/rocq/${rocq.version}/user-contrib/Transformer/
              cp -r Transformer/*.glob $out/lib/rocq/${rocq.version}/user-contrib/Transformer/
            '';

            meta = with pkgs.lib; {
              description = "Formally verified Transformer architecture in Rocq";
              license = licenses.mit;
              maintainers = [ ];
            };
          };
        };

        devShells.default = pkgs.mkShell {
          name = "rocq-transformer-dev";

          buildInputs = [
            rocq
            stdlib
          ];

          shellHook = ''
            echo "Rocq Transformer Development Environment"
            echo ""
            echo "Rocq version: $(rocq --version 2>/dev/null || coqc --version | head -1)"
            echo ""
            echo "Available commands:"
            echo "  make          - Build all modules"
            echo "  make clean    - Clean build artifacts"
            echo ""
          '';
        };

        # For `nix check`
        checks.default = self.packages.${system}.default;
      });
}
