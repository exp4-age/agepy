"""Processing and analysis of fluorescence spectra.

"""
from __future__ import annotations
from typing import Union, Tuple, Sequence
import warnings

import numpy as np
import h5py

from ._anodes import *


class Spectrum:
    """Fluorescence spectrum.

    """

    def __init__(self,
        raw: np.ndarray,
        time: int = None,
        anode: PositionAnode = None,
        target_density: float = None,
        intensity_downstream: float = None,
        intensity_upstream: float = None,
    ) -> None:
        # Store the passed data
        self._raw = raw
        self._t = time
        self.anode = anode
        self._target_density = target_density
        self._intensity_downstream = intensity_downstream
        self._intensity_upstream = intensity_upstream

    @classmethod
    def from_h5(cls,
        file_path: str,
        raw: str = "dld_rd#raw/0/0.0",
        time: int = None,
        anode: PositionAnode = None,
        target_density: str = None,
        intensity_downstream: str = None,
        intensity_upstream: str = None,
    ) -> None:
        with h5py.File(file_path, "r") as h5:
            # Load the raw data
            raw = np.asarray(h5[raw])
            # Load the target density
            if target_density is not None:
                target_density = h5[target_density][0]
                print("Target density:", target_density)
            # Load the upstream intensity
            if intensity_upstream is not None:
                intensity_upstream = h5[intensity_upstream][0]
            # Load the downstream intensity
            if intensity_downstream is not None:
                intensity_downstream = h5[intensity_downstream][0]
        # Initialize the Spectrum
        return cls(raw, time=time, anode=anode, target_density=target_density,
                   intensity_downstream=intensity_downstream,
                   intensity_upstream=intensity_upstream)

    @classmethod
    def from_scan(cls,
        file_path: str,
        scan_var: str,
        raw: str = "dld_rd#raw/0",
        time_per_step: int = None,
        anode: PositionAnode = None,
        target_density: str = None,
        intensity_downstream: str = None,
        intensity_upstream: str = None,
    ) -> Tuple[list, list]:
        with h5py.File(file_path, "r") as h5:
            # Load the steps
            steps = h5[scan_var]
            # Load the raw data
            raw = h5[raw]
            # Load the target density
            if target_density is not None:
                target_density = np.asarray(h5[target_density])
            else:
                target_density = np.full(len(steps), None)
            # Load upstream intensity
            if intensity_upstream is not None:
                intensity_upstream = np.asarray(h5[intensity_upstream])
            else:
                intensity_upstream = np.full(len(steps), None)
            # Load downstream intensity
            if intensity_downstream is not None:
                intensity_downstream = np.asarray(h5[intensity_downstream])
            else:
                intensity_downstream = np.full(len(steps), None)
            # Format the data and steps
            spectra = []
            step_val = []
            for i, step in enumerate(steps.keys()):
                # Format the step value
                step_val.append(steps[step][0][0])
                # Format the raw data
                data = np.asarray(raw[step])
                # Initialize the spectrum instance
                spectra.append(cls(
                    data,
                    time=time_per_step,
                    anode=anode,
                    target_density=target_density[i],
                    intensity_downstream=intensity_downstream[i],
                    intensity_upstream=intensity_upstream[i]
                ))
        # Return the spectra and energies
        return spectra, step_val

    def counts(self,
        anode: PositionAnode = None,
        roi: dict = None,
        efficiency_map: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        background: Spectrum = None,
    ) -> Tuple[float, float]:
        """Get number of counts in the spectrum and the estimated
        uncertainty.

        Parameters
        ----------
        anode: PositionAnode
            Anode object from `agepy.spec.photons`.
        """
        # Get the xy values
        if anode is None:
            anode = self.anode
        if anode is None:
            raise ValueError("Anode object must be provided for processing.")
        det_image = anode.process(self._raw)
        # Use the full detector if roi not provided
        if roi is not None:
            # Apply y roi filter
            det_image = det_image[det_image[:,1] > roi["y"]["min"]]
            det_image = det_image[det_image[:,1] < roi["y"]["max"]]
            # Apply x roi filter
            det_image = det_image[det_image[:,0] > roi["x"]["min"]]
            det_image = det_image[det_image[:,0] < roi["x"]["max"]]
        # Apply spatial detector efficiency correction
        if efficiency_map is not None:
            eff, xe, ye = efficiency_map
            x_inds = np.digitize(det_image[:,0], xe)
            y_inds = np.digitize(det_image[:,1], ye)
            efficiencies = eff[x_inds, y_inds]
            # Get the inverse of the efficiency
            inv_eff = 1 / efficiencies[efficiencies > 0]
        else:
            inv_eff = np.ones(det_image.shape[0])
        # Calculate the number of counts
        n = np.sum(inv_eff)
        err = np.sqrt(len(inv_eff)) * n / len(inv_eff)
        # Normalize data to measurement duration
        if self._t is not None:
            n /= self._t
            err /= self._t
        # Subtract background before further normalization
        if background is not None:
            bkg_counts, bkg_err = background.counts(anode, roi=roi,
                efficiency_map=efficiency_map)
            n -= bkg_counts
            n = max(n, 0)
            err = np.sqrt(err**2 + bkg_err**2)
        # Normalize data to account for beam intensity, gas
        # pressure, etc.
        if self._target_density is not None:
            n /= self._target_density
            err /= self._target_density
        if self._intensity_upstream is not None:
            n /= self._intensity_upstream
            err /= self._intensity_upstream
        return n, err

    def spectrum(self,
        edges: np.ndarray,
        anode: PositionAnode = None,
        roi: dict = None,
        efficiency_map: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        background: Spectrum = None,
        phem_calib: Tuple[float, float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        TODO
        ----
        - Implement handling of the uncertainties of the efficiency map
        - Background subtraction is very primitive and uncertainties are
          not propagated correctly
        """
        # Get the xy values
        if anode is None:
            anode = self.anode
        if anode is None:
            raise ValueError("Anode object must be provided for processing.")
        det_image = anode.process(self._raw)
        # Define the roi as the full detector if not provided
        if roi is None:
            roi = {"x": {"min": 0, "max": 1}, "y": {"min": 0, "max": 1}}
        # Apply y roi filter
        det_image = det_image[det_image[:,1] > roi["y"]["min"]]
        det_image = det_image[det_image[:,1] < roi["y"]["max"]]
        # Apply spatial detector efficiency correction
        if efficiency_map is not None:
            eff, xe, ye = efficiency_map
            x_inds = np.digitize(det_image[:,0], xe)
            y_inds = np.digitize(det_image[:,1], ye)
            efficiencies = eff[x_inds, y_inds]
        else:
            efficiencies = np.ones(det_image.shape[0])
        # Don't need y values anymore
        det_image = det_image[:,0].flatten()
        # Convert x values to wavelengths
        if phem_calib is not None:
            a, b = phem_calib
            det_image = a * det_image + b
            # Adjust x roi filter to wavelength binning
            wl_min = edges[edges < (a * roi["x"]["min"] + b)][-1]
            wl_max = edges[edges > (a * roi["x"]["max"] + b)][0]
            xroi = (det_image >= wl_min) & (det_image <= wl_max)
        else:
            # Adjust x roi filter to detector binning
            x_min = edges[edges < roi["x"]["min"]][-1]
            x_max = edges[edges > roi["x"]["max"]][0]
            xroi = (det_image >= x_min) & (det_image <= x_max)
        # Apply x roi filter
        det_image = det_image[xroi]
        efficiencies = efficiencies[xroi]
        # Calculate the sum of weights for each bin, i.e. the weighted spectrum
        spectrum = np.histogram(det_image, bins=edges, weights=efficiencies)[0]
        # Histogram data without the per event efficiencies
        errors = np.sqrt(np.histogram(det_image, bins=edges)[0])
        # Normalize data to measurement duration per step
        if self._t is not None:
            spectrum /= self._t
            errors /= self._t
        # Subtract background before further normalization
        if background is not None:
            bkg_spec, bkg_err = background.spectrum(
                edges, anode, roi=roi, efficiency_map=efficiency_map,
                phem_calib=phem_calib
            )
            spectrum -= bkg_spec
            spectrum[spectrum < 0] = 0
            errors = np.sqrt(errors**2 + bkg_err**2)
        # Normalize data to account for beam intensity, gas 
        # pressure, etc.
        if self._target_density is not None:
            spectrum /= self._target_density
            errors /= self._target_density
        if self._intensity_upstream is not None:
            spectrum /= self._intensity_upstream
            errors /= self._intensity_upstream
        # Return the spectrum and uncertainties
        return spectrum, errors


class EnergyScan:
    """Scan over exciting-photon energies.

    Parameters
    ----------
    data_files: Sequence[str]
        List of data files to be processed.
    energies: str
        Path to the exciting-photon energies in the data files.
    anode: PositionAnode
        Anode object from `agepy.spec.photons`.
    raw: str, optional
        Path to the raw data in the data files. Default:
        "dld_rd#raw/0".
    time_per_step: int, optional
        Time per step in the scan. Default: None.
    target_density: str, optional
        Path to the target density in the data files. Default: None.
    intensity_downstream: str, optional
        Path to the downstream intensity in the data files. Default:
        None.
    intensity_upstream: str, optional
        Path to the upstream intensity in the data files. Default:
        None.

    Attributes
    ----------
    anode: PositionAnode
        Anode object from `agepy.spec.photons`.
    spectra: np.ndarray
        Array of the loaded Spectrum objects.
    energies: np.ndarray
        Array of the exciting-photon energies.

    TODO
    ----
    - Very minimal implementation, needs to be expanded

    """

    def __init__(self,
        data_files: Sequence[str],
        energies: str,
        anode: PositionAnode,
        raw: str = "dld_rd#raw/0",
        time_per_step: int = None,
        target_density: str = None,
        intensity_downstream: str = None,
        intensity_upstream: str = None,
    ) -> None:
        self.anode = anode  
        self.spectra = []
        self.energies = []
        if isinstance(data_files, str):
            data_files = [data_files]
        for f in data_files:
            spec, E = Spectrum.from_scan(
                f, energies, raw=raw, time_per_step=time_per_step,
                anode=None, target_density=target_density,
                intensity_downstream=intensity_downstream,
                intensity_upstream=intensity_upstream
            )
            self.spectra.extend(spec)
            self.energies.extend(E)
        # Convert to numpy arrays
        self.energies = np.array(self.energies)
        self.spectra = np.array(self.spectra)
        # Sort the spectra by energy
        energy_sort = np.argsort(self.energies)
        self.energies = self.energies[energy_sort]
        self.spectra = self.spectra[energy_sort]

    def counts(self,
        roi: dict = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the photon-excitation energy spectrum.

        Parameters
        ----------
        roi: dict
            Region of interest for the detector. If not provided, the
            full detector is used.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The number of counts (normalized), the respective
            statistical uncertainties, and the exciting-photon energies.

        """
        vectorized_counts = np.vectorize(
            lambda spec: spec.counts(self.anode, roi=roi)
        )
        n, err = vectorized_counts(self.spectra)
        return n, err, self.energies

