import { registerComponent, PluginComponentType } from "@fiftyone/plugins";
import { useOperatorExecutor } from "@fiftyone/operators";
import React, { useState, useEffect } from "react";
import {
  Stack,
  Box,
  Button,
  Typography,
  Paper,
  Link,
  IconButton,
  Tooltip,
} from "@mui/material";
import { useRecoilValue, useSetRecoilState } from "recoil";
import { wandbURLAtom, reportModeAtom } from "./State";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import CloseIcon from "@mui/icons-material/Close";
import "./Operator";

export const WandBIcon = ({ size = "1rem", style = {} }) => {
  return (
    <img
      src="https://raw.githubusercontent.com/harpreetsahota204/fiftyone_wandb_plugin/refs/heads/main/assets/wandb.svg"
      alt="W&B Logo"
      height={size}
      width={size}
      style={style}
    />
  );
};

export default function WandBPanel() {
  const defaultUrl = "https://wandb.ai";
  const wandbUrl = useRecoilValue(wandbURLAtom);
  const reportMode = useRecoilValue(reportModeAtom);
  const setReportMode = useSetRecoilState(reportModeAtom);
  
  // Memoize urlToOpen to prevent unnecessary recalculations
  const urlToOpen = React.useMemo(() => wandbUrl || defaultUrl, [wandbUrl, defaultUrl]);

  const handleOpenDashboard = React.useCallback(() => {
    window.open(urlToOpen, "_blank", "noopener,noreferrer");
  }, [urlToOpen]);

  const handleCloseReport = React.useCallback(() => {
    setReportMode(false);
  }, [setReportMode]);

  // If in report mode, show iframe
  if (reportMode) {
    return (
      <Stack
        sx={{
          width: "100%",
          height: "100%",
          position: "relative",
        }}
      >
        <Box
          sx={{
            position: "absolute",
            top: 8,
            right: 8,
            zIndex: 1000,
            backgroundColor: "rgba(255, 255, 255, 0.9)",
            borderRadius: 1,
          }}
        >
          <Tooltip title="Close embedded report">
            <IconButton onClick={handleCloseReport} size="small">
              <CloseIcon />
            </IconButton>
          </Tooltip>
        </Box>
        <Box
          sx={{
            width: "100%",
            height: "100%",
            overflow: "hidden",
          }}
        >
          <iframe
            key={urlToOpen}
            style={{
              width: "100%",
              height: "100%",
              border: "none",
            }}
            src={urlToOpen}
            title="W&B Report"
            allowFullScreen
          />
        </Box>
      </Stack>
    );
  }

  // Otherwise show launcher mode
  return (
    <Stack
      sx={{
        width: "100%",
        height: "100%",
        alignItems: "center",
        justifyContent: "center",
        padding: 4,
      }}
      spacing={3}
    >
      <Paper
        elevation={3}
        sx={{
          padding: 4,
          maxWidth: 600,
          textAlign: "center",
        }}
      >
        <Box sx={{ marginBottom: 3 }}>
          <WandBIcon size="4rem" />
        </Box>
        
        <Typography variant="h5" gutterBottom>
          Weights & Biases Dashboard
        </Typography>
        
        <Typography variant="body1" color="text.secondary" paragraph>
          Click the button below to open your W&B dashboard in a new browser tab.
        </Typography>

        <Typography variant="body2" color="text.secondary" paragraph>
          Current URL: <Link href={urlToOpen} target="_blank" rel="noopener noreferrer">{urlToOpen}</Link>
        </Typography>

        <Button
          variant="contained"
          size="large"
          onClick={handleOpenDashboard}
          endIcon={<OpenInNewIcon />}
          sx={{ marginTop: 2 }}
        >
          Open W&B Dashboard
        </Button>

        <Typography variant="caption" display="block" sx={{ marginTop: 3 }} color="text.secondary">
          Use "Show W&B Run" to open runs or "Show W&B Report" to embed reports here.
        </Typography>
      </Paper>
    </Stack>
  );
}

registerComponent({
  name: "WandBPanel",
  label: "Weights & Biases Dashboard",
  component: WandBPanel,
  type: PluginComponentType.Panel,
  Icon: () => <WandBIcon size={"1rem"} style={{ marginRight: "0.5rem" }} />,
});

