import React from 'react';
export const metadata = {
    title: 'Qwen3.5-4B in 3D',
    description: 'A 3D walkthrough of one real forward pass of Qwen3.5-4B.',
};
export default function RootLayout({ children }: { children: React.ReactNode }) {
    return <html lang="en"><body>{children}</body></html>;
}
